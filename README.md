[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/lmppl/blob/master/LICENSE.txt)
[![PyPI version](https://badge.fury.io/py/lmppl.svg)](https://badge.fury.io/py/lmppl)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/lmppl.svg)](https://pypi.python.org/pypi/lmppl/)
[![PyPI status](https://img.shields.io/pypi/status/lmppl.svg)](https://pypi.python.org/pypi/lmppl/)

# Language Model Perplexity (LM-PPL) 
Perplexity measures how predictable a text is by a language model (LM), and it is often used to evaluate fluency or proto-typicality of the text
(lower the perplexity is, more fluent or proto-typical the text is).
LM-PPL is a python library to calculate perplexity on a text with any types of pre-trained LMs.
We compute an ordinary perplexity for recurrent LMs such as [GPT3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165) and the perplexity of the decoder for encoder-decoder 
LMs such as [BART (Lewis et al., 2020)](https://aclanthology.org/2020.acl-main.703/) or [T5 (Raffel et al., 2020)](https://arxiv.org/abs/1910.10683), 
while we compute [pseudo-perplexity (Wang and Cho, 2018)](https://aclanthology.org/W19-2304/) for masked LMs. 


## Get Started
Install via pip.
```shell
pip install lmppl
```

### Example 
Let's solve sentiment analysis with perplexity as an example! Remember the text with lower perplexity is better, so we 
compare two texts (positive and negative) and choose the one with lower perplexity as the model prediction.


1. ***Recurrent LM*** including variants of GPT.
```python3
import lmppl

scorer = lmppl.LM('gpt2')
text = [
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad.'
]
ppl = scorer.get_perplexity(text)
print(list(zip(text, ppl)))
>>> [
  ('sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy.', 136.64255272925908),
  ('sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad.', 139.2400838400971)
]
print(f"prediction: {text[ppl.index(min(ppl))]}")
>>> "prediction: sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy."
```

2. ***Masked LM*** including variants of BERT.
```python3
import lmppl

scorer = lmppl.MaskedLM('microsoft/deberta-v3-small')
text = [
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad.'
]
ppl = scorer.get_perplexity(text)
print(list(zip(text, ppl)))
>>> [
  ('sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy.', 1190212.1699246117),
  ('sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad.', 1152767.482071837)
]
print(f"prediction: {text[ppl.index(min(ppl))]}")
>>> "prediction: sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad."
```


3. ***Encoder-Decoder LM*** including variants of T5 and BART.
```python3
import lmppl

scorer = lmppl.EncoderDecoderLM('google/flan-t5-small')
inputs = [
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee.',
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee.'
]
outputs = [
    'I am happy.',
    'I am sad.'
]
ppl = scorer.get_perplexity(input_texts=inputs, output_texts=outputs)
print(list(zip(outputs, ppl)))
>>> [
  ('I am happy.', 4138.748977714201),
  ('I am sad.', 2991.629250051472)
]
print(f"prediction: {outputs[ppl.index(min(ppl))]}")
>>> "prediction: I am sad."
```

### Tips
- **Max Token Length**: Each LM has its own max-token length (`max_length` for recurrent/masked LMs, and `max_length_encoder` and `max_length_decoder` for encoder-decoder LMs).
Limiting those max-token will reduce the time to process the text, but it may affect the accuracy of the perplexity, so please experiment on your texts and decide
an optimal token length.
  
- **Batch Size**: One can pass batch size to the function `get_perplexity` (eg. `get_perplexity(text, batch_size=32)`).
As default, it will process all the text once, that may cause memory error if the number of texts is too large.

