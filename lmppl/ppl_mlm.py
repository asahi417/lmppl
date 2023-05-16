""" Caluculate MLM pseudo perplexity.
>>> scorer = MaskedLM()
>>> scores = scorer.get_perplexity(
    input_texts=['sentiment classification: I have a bad day is happy',
                 'sentiment classification: I have a bad day is sad'],
)
>>> print(scores)
[128.80070356559577, 100.5730992106926]
"""
import os
import logging
import math
from typing import List
from tqdm import tqdm
from itertools import chain

import transformers
import torch

from .util import internet_connection

os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index


class MaskedLM:
    """ Masked Language Model. """

    def __init__(self,
                 model: str = 'distilbert-base-uncased',
                 use_auth_token: bool = False,
                 max_length: int = None,
                 num_gpus: int = None,
                 torch_dtype=None,
                 device_map: str = None,
                 low_cpu_mem_usage: bool = False,
                 trust_remote_code: bool = True,
                 offload_folder: str = None,
                 hf_cache_dir: str = None):
        """ Masked Language Model.

        @param model: Model alias or path to local model file.
        @param use_auth_token: Huggingface transformers argument of `use_auth_token`
        @param device: Device name to load the models.
        @param num_gpus: Number of gpus to be used.
        """
        logging.info(f'Loading Model: `{model}`')

        # load model
        params = {"local_files_only": not internet_connection(), "use_auth_token": use_auth_token,
                  "trust_remote_code": trust_remote_code}
        if hf_cache_dir is not None:
            params["cache_dir"] = hf_cache_dir
        if offload_folder is not None:
            params["offload_folder"] = offload_folder
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, **params)
        self.config = transformers.AutoConfig.from_pretrained(model, **params)
        params.update({"config": self.config, "low_cpu_mem_usage": low_cpu_mem_usage})
        if torch_dtype is not None:
            params['torch_dtype'] = torch_dtype
        if device_map is not None:
            params['device_map'] = device_map
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(model, **params)
        if max_length is None:
            self.max_length = None
        else:
            self.max_length = max_length if max_length is not None else self.tokenizer.model_max_length
            assert self.max_length <= self.tokenizer.model_max_length, f"{self.max_length} > {self.tokenizer.model_max_length}"

        # loss function
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        # search sentence-prefix tokens
        tokens = self.tokenizer.tokenize('get tokenizer specific prefix')
        tokens_encode = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode('get tokenizer specific prefix'))
        self.sp_token_prefix = tokens_encode[:tokens_encode.index(tokens[0])]
        self.sp_token_suffix = tokens_encode[tokens_encode.index(tokens[-1]) + 1:]
        self.mask_token = self.tokenizer.mask_token

        # GPU setup
        self.device = self.model.device
        if device_map is None:
            num_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus
            if num_gpus == 1:
                self.model.to('cuda')
                self.device = self.model.device
            elif num_gpus > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.model.to('cuda')
                self.device = self.model.module.device
        self.model.eval()
        logging.info(f'\t * model is loaded on: {self.device}')

    def get_perplexity(self, input_texts: str or List, batch: int = None):
        """ Compute the perplexity on MLM.

        :param input_texts: A string or list of input texts for the encoder.
        :param batch: Batch size
        :return: A value or list of perplexity.
        """

        single_input = type(input_texts) == str
        input_texts = [input_texts] if single_input else input_texts

        def get_partition(_list):
            length = list(map(lambda o: len(o), _list))
            return list(map(lambda o: [sum(length[:o]), sum(length[:o + 1])], range(len(length))))

        # data preprocessing
        data = []
        for x in input_texts:
            x = self.tokenizer.tokenize(x)

            def encode_mask(mask_position: int):
                _x = x.copy()
                # get the token id of the correct token
                masked_token_id = self.tokenizer.convert_tokens_to_ids(_x[mask_position])
                # mask the token position
                _x[mask_position] = self.tokenizer.mask_token
                # convert into a sentence
                _sentence = self.tokenizer.convert_tokens_to_string(_x)
                # encode
                if self.max_length is not None:
                    _e = self.tokenizer(
                        _sentence, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
                else:
                    _e = self.tokenizer(_sentence, truncation=True, padding=True, return_tensors='pt')
                # add the correct token id as the label
                label = [PAD_TOKEN_LABEL_ID] * _e['input_ids'].shape[1]
                label[mask_position + len(self.sp_token_prefix)] = masked_token_id
                _e['labels'] = torch.tensor([label], dtype=torch.long)
                return _e

            if self.max_length is not None:
                data.append([encode_mask(i) for i in range(min(self.max_length - len(self.sp_token_prefix), len(x)))])
            else:
                data.append([encode_mask(i) for i in range(len(x))])

        # get partition
        partition = get_partition(data)
        data = list(chain(*data))

        # batch preparation
        batch = len(data) if batch is None else batch
        batch_id = list(range(0, len(data), batch)) + [len(data)]
        batch_id = list(zip(batch_id[:-1], batch_id[1:]))

        # run model
        nll = []
        with torch.no_grad():
            for s, e in tqdm(batch_id):
                _encode = data[s:e]
                _encode = {k: torch.cat([o[k] for o in _encode], dim=0).to(self.device) for k in _encode[0].keys()}
                labels = _encode.pop('labels')
                output = self.model(**_encode, return_dict=True)
                prediction_scores = output['logits']
                loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
                loss = loss.view(len(prediction_scores), -1)
                loss = torch.sum(loss, -1)
                nll += loss.cpu().tolist()

        # reconstruct the nested structure
        ppl = list(map(lambda o: math.exp(sum(nll[o[0]:o[1]]) / (o[1] - o[0])), partition))
        if single_input:
            return ppl[0]
        return ppl

