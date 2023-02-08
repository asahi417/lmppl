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
from math import exp
from typing import List

from tqdm import tqdm
import torch
import transformers

from .util import internet_connection

os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message


def get_lm(model_name: str, use_auth_token: bool = False):
    """ get encoder-decoder lms from huggingface """
    # tokenizer
    local_files_only = not internet_connection()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only, use_auth_token=use_auth_token)

    # config
    config = transformers.AutoConfig.from_pretrained(
        model_name, local_files_only=local_files_only, use_auth_token=use_auth_token)

    # model
    if config.model_type == 't5':  # T5 model requires T5ForConditionalGeneration class
        model_class = transformers.T5ForConditionalGeneration.from_pretrained
    elif config.model_type == 'mt5':
        model_class = transformers.MT5ForConditionalGeneration.from_pretrained
    elif config.model_type == 'bart':
        model_class = transformers.BartForConditionalGeneration.from_pretrained
    elif config.model_type == 'mbart':
        model_class = transformers.MBartForConditionalGeneration.from_pretrained
    else:
        raise ValueError(f'unsupported model type: {config.model_type}')
    model = model_class(
        model_name, config=config, local_files_only=local_files_only, use_auth_token=use_auth_token)

    # add new special tokens to the tokenizer and the model if they don't have it
    return tokenizer, model, config


class EncoderDecoderLM:
    """ Encoder-Decoder Language Model """

    def __init__(self,
                 model: str = 't5-small',
                 use_auth_token: bool = False,
                 max_length_encoder: int = None,
                 max_length_decoder: int = None,
                 device: str = None,
                 num_gpus: int = None):
        """ Encoder-Decoder Language Model.

        @param model: Model alias or path to local model file.
        @param use_auth_token: Huggingface transformers argument of `use_auth_token`
        @param device: Device name to load the models.
        @param num_gpus: Number of gpus to be used.
        """
        logging.info(f'Loading Model: `{model}`')

        # load model
        self.tokenizer, self.model, self.config = get_lm(model, use_auth_token=use_auth_token)
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
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction='none')

        # GPU setup
        if device is None:
            self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        else:
            self.device = device
        num_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus
        if num_gpus > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()
        logging.info(f'\t * Num of GPU in use: {torch.cuda.device_count()}')

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
                with self.tokenizer.as_target_tokenizer():
                    if self.max_length_encoder is not None:
                        labels = self.tokenizer(output_texts[s:e], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length_decoder)
                    else:
                        labels = self.tokenizer(output_texts[s:e], return_tensors='pt', padding=True, truncation=True)
                    model_inputs["labels"] = labels["input_ids"]

                # model run & loss conversion into likelihood
                out = self.model(**{k: v.to(self.device) for k, v in model_inputs.items()})
                loss = self.loss_fct(
                    out['logits'].view(-1, out['logits'].size(-1)),
                    model_inputs["labels"].view(-1).to(self.device)
                )
                loss_aligned = loss.view(out['logits'].size(0), out['logits'].size(1))
                loss_final = loss_aligned.mean(-1)

                # add to the list
                loss_list += loss_final.cpu().numpy().tolist()

        # conversion to perplexity
        ppl = [exp(i) for i in loss_list]

        assert len(ppl) == len(input_texts), f"{len(ppl)} != {len(input_texts)}"
        return ppl[0] if single_input else ppl
