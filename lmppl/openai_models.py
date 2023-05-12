""" Calculate perplexity.
>>> scorer = OpenAI(api_key="your-api-key", model="gpt-4-32k")
>>> scores = scorer.get_perplexity(
    input_texts=['sentiment classification: I have a bad day is happy',
                 'sentiment classification: I have a bad day is sad'],
)
>>> print(scores)
[128.80070356559577, 100.5730992106926]
"""

import logging
from math import exp
from typing import List
from tqdm import tqdm
from statistics import mean
from time import sleep
import openai


class OpenAI:
    """ Language Model. """

    def __init__(self, api_key: str, model: str, sleep_time: int = 10):
        """ Language Model.

        @param api_key: OpenAI API key.
        @param model: OpenAI model.
        """
        logging.info(f'Loading Model: `{model}`')
        openai.api_key = api_key
        self.model = model
        self.sleep_time = sleep_time

    def get_perplexity(self, input_texts: str or List, *args, **kwargs):
        """ Compute the perplexity on recurrent LM.

        :param input_texts: A string or list of input texts for the encoder.
        :return: A value or list of perplexity.
        """
        single_input = type(input_texts) == str
        input_texts = [input_texts] if single_input else input_texts
        nll = []
        for text in tqdm(input_texts):
            # https://platform.openai.com/docs/api-reference/completions/create
            while True:
                try:
                    completion = openai.Completion.create(
                        model=self.model,
                        prompt=text,
                        logprobs=0, # Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. https://platform.openai.com/docs/api-reference/completions/create#completions/create-logprobs
                        max_tokens=0,
                        temperature=1.0,
                        echo=True  # Echo back the prompt in addition to the completion https://platform.openai.com/docs/api-reference/completions/create#completions/create-echo
                    )
                    break
                except Exception:
                # except openai.error.RateLimitError:
                    if self.sleep_time is None or self.sleep_time == 0:
                        logging.exception('OpenAI internal error')
                        exit()
                    logging.info(f'Rate limit exceeded. Waiting for {self.sleep_time} seconds.')
                    sleep(self.sleep_time)
            nll.append(mean([i for i in completion['choices'][0]['logprobs']['token_logprobs'] if i is not None]))
        ppl = [exp(-i) for i in nll]
        return ppl[0] if single_input else ppl
