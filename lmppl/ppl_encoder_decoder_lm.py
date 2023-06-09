""" Caluculate decoder perpleity of encoder-decoder LM.
>>> from lmppl import EncoderDecoderLM
>>> scorer = EncoderDecoderLM('t5-small')
>>> scores = scorer.get_perplexity(
        input_texts=['sentiment classification: I have a bad day'] * 2,
        output_texts=['happy', 'sad'])
>>> print(scores)
[373.821367795063, 274.29454188096724]
"""
import os
import logging
import gc
from math import exp
from typing import List

from tqdm import tqdm
import torch
import transformers

from .util import internet_connection

os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
FORCE_RESET = bool(int(os.getenv("FORCE_RESET", "0")))


def get_lm(model_name: str,
           use_auth_token: bool = False,
           torch_dtype=None,
           device_map: str = None,
           low_cpu_mem_usage: bool = False,
           trust_remote_code: bool = True,
           offload_folder: str = None,
           hf_cache_dir: str = None):
    """ get encoder-decoder lms from huggingface """
    # tokenizer
    params = {"local_files_only": not internet_connection(), "use_auth_token": use_auth_token,
              "trust_remote_code": trust_remote_code}
    if hf_cache_dir is not None:
        params["cache_dir"] = hf_cache_dir
    if offload_folder is not None:
        params["offload_folder"] = offload_folder
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **params)

    # config
    config = transformers.AutoConfig.from_pretrained(model_name, **params)

    # model
    if config.model_type == 't5':  # T5 model requires T5ForConditionalGeneration class
        model_class = transformers.T5ForConditionalGeneration.from_pretrained
    elif config.model_type == 'mt5':
        model_class = transformers.MT5ForConditionalGeneration.from_pretrained
    elif config.model_type == 'bart':
        model_class = transformers.BartForConditionalGeneration.from_pretrained
    elif config.model_type == 'mbart':
        model_class = transformers.MBartForConditionalGeneration.from_pretrained
    elif config.model_type == 'switch_transformers':
        model_class = transformers.SwitchTransformersForConditionalGeneration.from_pretrained
    else:
        raise ValueError(f'unsupported model type: {config.model_type}')
    params.update({'config': config, "low_cpu_mem_usage": low_cpu_mem_usage})
    if torch_dtype is not None:
        params['torch_dtype'] = torch_dtype
    if device_map is not None:
        params['device_map'] = device_map
    model = model_class(model_name, **params)
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    return tokenizer, model, config


class EncoderDecoderLM:
    """ Encoder-Decoder Language Model """

    def __init__(self,
                 model: str = 't5-small',
                 use_auth_token: bool = False,
                 max_length_encoder: int = None,
                 max_length_decoder: int = None,
                 num_gpus: int = None,
                 torch_dtype=None,
                 device_map: str = None,
                 low_cpu_mem_usage: bool = False,
                 trust_remote_code: bool = True,
                 offload_folder: str = None,
                 hf_cache_dir: str = None):
        """ Encoder-Decoder Language Model.

        @param model: Model alias or path to local model file.
        @param use_auth_token: Huggingface transformers argument of `use_auth_token`
        @param device: Device name to load the models.
        @param num_gpus: Number of gpus to be used.
        """
        logging.info(f'Loading Model: `{model}`')

        # load model
        self.device_map = device_map
        self.tokenizer, self.model, self.config = get_lm(
            model, use_auth_token=use_auth_token, torch_dtype=torch_dtype, device_map=self.device_map,
            low_cpu_mem_usage=low_cpu_mem_usage, hf_cache_dir=hf_cache_dir, trust_remote_code=trust_remote_code,
            offload_folder=offload_folder)

        self.pad_token_initialized = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': "<<PAD>>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.pad_token_initialized = True

        if max_length_encoder is None:
            self.max_length_encoder = None
        else:
            self.max_length_encoder = max_length_encoder if max_length_encoder is not None else self.tokenizer.model_max_length
            assert self.max_length_encoder <= self.tokenizer.model_max_length, f"{self.max_length_encoder} > {self.tokenizer.model_max_length}"
        if max_length_decoder is None:
            self.max_length_decoder = None
        else:
            self.max_length_decoder = max_length_decoder if max_length_decoder is not None else self.tokenizer.model_max_length
            assert self.max_length_decoder <= self.tokenizer.model_max_length, f"{self.max_length_decoder} > {self.tokenizer.model_max_length}"

        # loss function
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        # GPU setup
        self.device = self.model.device
        if self.device_map is None:
            num_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus
            if num_gpus == 1:
                self.model.cuda()
                self.device = self.model.device
            elif num_gpus > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.model.cuda()
                self.device = self.model.module.device
        self.model.eval()
        logging.info(f'\t * model is loaded on: {self.device}')

    def get_perplexity(self, input_texts: str or List, output_texts: str or List, batch: int = None):
        """ Compute the perplexity on decoder of the seq2seq model.

        :param input_texts: A string or list of input texts for the encoder.
        :param output_texts: A string or list of output texts for the decoder.
        :param batch: Batch size
        :return: A value or list of perplexity.
        """
        assert type(input_texts) is type(output_texts), f"{type(input_texts)} != {type(output_texts)}"

        # batch preparation
        single_input = type(input_texts) == str
        input_texts = [input_texts] if single_input else input_texts
        output_texts = [output_texts] if single_input else output_texts
        assert len(input_texts) == len(output_texts), f"{len(input_texts)} == {len(output_texts)}"
        batch = len(output_texts) if batch is None else batch
        batch_id = list(range(0, len(input_texts), batch)) + [len(output_texts)]
        batch_id = list(zip(batch_id[:-1], batch_id[1:]))

        loss_list = []
        with torch.no_grad():
            for s, e in tqdm(batch_id):

                # input feature
                if self.max_length_encoder is not None:
                    model_inputs = self.tokenizer(
                        input_texts[s:e], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length_encoder)
                else:
                    model_inputs = self.tokenizer(input_texts[s:e], return_tensors='pt', padding=True, truncation=True)

                if self.max_length_decoder is not None:
                    output_encode = self.tokenizer(text_target=output_texts[s:e], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length_decoder)
                else:
                    output_encode = self.tokenizer(text_target=output_texts[s:e], return_tensors='pt', padding=True, truncation=True)

                # shift the label sequence for causal inference
                label = output_encode["input_ids"]
                label[label == self.tokenizer.pad_token_id] = PAD_TOKEN_LABEL_ID

                if self.device_map is None:
                    model_inputs["labels"] = label.to(self.device)
                    output = self.model(**{k: v.to(self.device) for k, v in model_inputs.items()})
                else:
                    model_inputs["labels"] = label
                    output = self.model(**{k: v.cuda() for k, v in model_inputs.items()})
                    model_inputs["labels"] = label.to(self.device)

                # model run & loss conversion into likelihood
                logits = output['logits']
                if self.pad_token_initialized:
                    logits = logits[:, :, :-1]
                valid_length = (model_inputs["labels"] != PAD_TOKEN_LABEL_ID).sum(dim=-1)
                loss = self.loss_fct(logits.view(-1, self.config.vocab_size), model_inputs["labels"].view(-1))
                loss = loss.view(len(logits), -1)
                loss = torch.sum(loss, -1) / valid_length
                loss_list += loss.cpu().tolist()

                if FORCE_RESET:
                    del model_inputs
                    del loss
                    del output
                    gc.collect()
                    torch.cuda.empty_cache()

        # conversion to perplexity
        ppl = [exp(i) for i in loss_list]
        return ppl[0] if single_input else ppl


if __name__ == '__main__':

    # scorer = LM("gpt2")
    scorer = EncoderDecoderLM("t5-small")
    _x = 'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee.'
    print(scorer.get_perplexity(input_texts=[_x] * 2, output_texts=['I am happy.', 'I am sad.']))

