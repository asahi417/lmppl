import json
import os
from statistics import mean
from glob import glob
import pandas as pd
from itertools import permutations

full_accuracy = []
for i in glob("metaphor_results/scores/*.json"):

    data_name = os.path.basename(i).split('.')[-2]
    model_name = ".".join(os.path.basename(i).split('.')[:-2])

    with open(i, "r") as f:
        data = json.load(f)

    num_option = len(data['labels'][0])
    metaphor_score = pd.DataFrame([i['score'].values for _, i in pd.DataFrame(data['metaphor']).groupby("index")],
                                  columns=list(range(num_option))).T
    literal_score = pd.DataFrame([i['score'].values for _, i in pd.DataFrame(data['literal']).groupby("index")],
                                 columns=list(range(num_option))).T
    anomaly_score = pd.DataFrame([i['score'].values for _, i in pd.DataFrame(data['anomaly']).groupby("index")],
                                 columns=list(range(num_option))).T
    accuracy_metaphor = []
    accuracy_literal = []
    accuracy_anomaly = []
    for n, label in enumerate(data['labels']):

        if num_option == 2:
            accuracy_metaphor.append(int(data["answer"][n] == metaphor_score[n].values.tolist().index(min(metaphor_score[n].values.tolist()))))
        else:
            accuracy_metaphor.append(int(data["answer"][n] == metaphor_score[n].values.tolist().index(min(metaphor_score[n].values.tolist()))))
            accuracy_literal.append(int(data["answer"][n] == literal_score[n].values.tolist().index(min(literal_score[n].values.tolist()))))
            accuracy_anomaly.append(int(data["answer"][n] == anomaly_score[n].values.tolist().index(min(anomaly_score[n].values.tolist()))))
            # true = "-".join([str(i) for i in label])
            # pred = {}
            # for opt in permutations(range(num_option), num_option):
            #     a, b, c = opt
            #     pred[f"{a}-{b}-{c}"] = mean([metaphor_score[n][a], anomaly_score[n][b], literal_score[n][c]])
            # accuracy.append(int(pred[true] == min(pred.values())))
    full_accuracy.append({"data": data_name, "label_type": "metaphor", "model": model_name, "accuracy": mean(accuracy_metaphor) * 100})
    if len(accuracy_literal) != 0:
        full_accuracy.append({"data": data_name, "label_type": "literal", "model": model_name, "accuracy": mean(accuracy_literal) * 100})
    if len(accuracy_anomaly) != 0:
        full_accuracy.append({"data": data_name, "label_type": "anomaly", "model": model_name, "accuracy": mean(accuracy_anomaly) * 100})

df = pd.DataFrame(full_accuracy).sort_values(by=['data', "model"])
df.to_csv("metaphor_results/result.csv", index=False)