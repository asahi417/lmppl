""" An example of solving analogy question https://huggingface.co/datasets/relbert/analogy_questions in zero-shot manner """

from itertools import chain
from statistics import mean
from datasets import load_dataset
from lmppl import LM

# load model and prompt the candidate and query
dataset = load_dataset('relbert/analogy_questions', 'sat', split='test')
template = "<subj-a> is to <obj-a> what <subj-b> is to <obj-b>"
choices = []
answers = []
for n, i in enumerate(dataset):
    a, b = i['stem']
    choices.append([template.replace('<subj-a>', a).replace('<obj-a>', b).replace('<subj-b>', x).replace('<obj-b>', y) for x, y in i['choice']])
    answers.append(i['answer'])

# flatten data
flatten_choice = list(chain(*choices))
flatten_index = list(chain(*[[n] * len(x) for n, x in enumerate(choices)]))

# get perplexity
model = LM('gpt2-xl')
scores = model.get_perplexity(flatten_choice, batch=32)

# reconstruct the data structure and compute accuracy
index_score = list(zip(flatten_index, scores))
index_score_nested = [[s for i, s in index_score if x == i] for x in sorted(list(set(flatten_index)))]
prediction = [i.index(min(i)) for i in index_score_nested]
accuracy = mean([a == b for a, b in zip(prediction, answers)])
print(accuracy)
