import lmppl


print("### MLM ###")
scorer = lmppl.MaskedLM('microsoft/deberta-v3-small')
text = [
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad.'
]
ppl = scorer.get_perplexity(text)
print(list(zip(text, ppl)))
print(f"prediction: {text[ppl.index(min(ppl))]}")


print("### LM ###")
scorer = lmppl.LM('gpt2')
text = [
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad.'
]
ppl = scorer.get_perplexity(text)
print(list(zip(text, ppl)))
print(f"prediction: {text[ppl.index(min(ppl))]}")


print("### Encoder-Decoder LM ###")
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
print(f"prediction: {outputs[ppl.index(min(ppl))]}")

print("### LM ###")
scorer = lmppl.LM('gpt2', device_map="auto", low_cpu_mem_usage=True)
text = [
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
    'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad.'
]
ppl = scorer.get_perplexity(text)
print(list(zip(text, ppl)))
print(f"prediction: {text[ppl.index(min(ppl))]}")


print("### Encoder-Decoder LM ###")
scorer = lmppl.EncoderDecoderLM('google/flan-t5-small', device_map="auto", low_cpu_mem_usage=True)
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
