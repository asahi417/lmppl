from itertools import chain
from statistics import mean
from datasets import load_dataset
from lmppl import LM

template = "<subj-a> is to <obj-a> what <subj-b> is to <obj-b>"
dataset = load_dataset('relbert/analogy_questions', 'sat', split='test')

choices = []
answers = []
for i in dataset:
    a, b = i['stem']
    choices.append([template.replace('<subj-a>', a).replace('<obj-a>', b).replace('<subj-b>', x).replace('<obj-b>', y) for x, y in i['choice']])
    answers.append(i['answer'])

flatten_choice = list(chain(*choices))
flatten_index = list(chain(*[[n] * len(x) for n, x in enumerate(choices)]))


model = LM('gpt2-xl')
scores = model.get_perplexity(flatten_choice, batch=32)
index_score = list(zip(flatten_index, scores))
index_score_nested = [[s for i, s in index_score if x == i] for x in sorted(list(set(flatten_index)))]
prediction = [i.index(min(i)) for i in index_score_nested]
accuracy = mean([a == b for a, b in zip(prediction, answers)])
print(accuracy)