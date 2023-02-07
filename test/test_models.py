import lmppl

print("### LM ###")
scorer = lmppl.LM('distilgpt2', max_length=64, )
text = [
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad.'
]
ppl = scorer.get_perplexity(text)
print(list(zip(text, ppl)))
print(f"prediction: {text[ppl.index(min(ppl))]}")

print("### MLM ###")
scorer = lmppl.MaskedLM('distilbert-base-uncased', max_length=64)
text = [
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad.'
]
ppl = scorer.get_perplexity(text)
print(list(zip(text, ppl)))
print(f"prediction: {text[ppl.index(min(ppl))]}")

print("### Encoder-Decoder LM ###")
scorer = lmppl.EncoderDecoderLM('google/flan-t5-small', max_length_encoder=64, max_length_decoder=16)
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
print(f"prediction: {outputs[ppl.index(min(ppl))]}")
