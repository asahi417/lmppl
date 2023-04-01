""" Solving Metaphor Detection via Prompting """
import json
import ast
import logging
import os
import gc
from typing import List
import torch
import lmppl
from datasets import load_dataset
from string import ascii_letters

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
label_template = {"metaphor": "is a metaphor", "literal": "is literal", "anomaly": "is difficult to interpret"}
template_header = "Which of the following examples"
template_footer = "The answer is "


def prompt(options: List, separate_in_out: bool, is_sentence: bool = False):
    if not is_sentence:
        assert all(len(i) == 4 for i in options), options
        statement = '\n'.join([f'{ascii_letters[n]}) {i[0]} is to {i[1]} what {i[2]} is to {i[3]}' for n, i in enumerate(options)])
    else:
        statement = '\n'.join([f'{ascii_letters[n]}) {i}' for n, i in enumerate(options)])
    text_input_m = f"{template_header} {label_template['metaphor']}?\n{statement}\n{template_footer}"
    text_input_l = f"{template_header} {label_template['literal']}?\n{statement}\n{template_footer}"
    text_input_a = f"{template_header} {label_template['anomaly']}?\n{statement}\n{template_footer}"
    if separate_in_out:
        return [[
            [text_input_m, ascii_letters[n]],
            [text_input_l, ascii_letters[n]],
            [text_input_a, ascii_letters[n]]
        ] for n in range(len(options))]
    return [[
        f"{text_input_m}\n{ascii_letters[n]}",
        f"{text_input_l}\n{ascii_letters[n]}",
        f"{text_input_a}\n{ascii_letters[n]}"
    ] for n in range(len(options))]


dataset_list = [  # dataset, dataset_name, split
    ['Joanne/Metaphors_and_Analogies', "Quadruples_Green_set", "test"],
    ['Joanne/Metaphors_and_Analogies', 'Pairs_Cardillo_set', "test"],
    ['Joanne/Metaphors_and_Analogies', 'Pairs_Jankowiac_set', "test"],
    # ["Joanne/katz1980_set_A", None, "test"]
]

language_models = {
    "google/flan-ul2": [lmppl.EncoderDecoderLM, 1],  # 20B
    "google/flan-t5-xxl": [lmppl.EncoderDecoderLM, 1],  # 11B
    "google/flan-t5-xl": [lmppl.EncoderDecoderLM, 4],  # 3B
    "google/flan-t5-large": [lmppl.EncoderDecoderLM, 256],  # 770M
    "google/flan-t5-base": [lmppl.EncoderDecoderLM, 1024],  # 220M
    "google/flan-t5-small": [lmppl.EncoderDecoderLM, 1024],  # 60M
    "facebook/opt-iml-30b": [lmppl.LM, 1],  # 30B
    "facebook/opt-iml-max-30b": [lmppl.LM, 1],  # 30B
    "facebook/opt-iml-max-1.3b": [lmppl.LM, 8],  # 1.3B
    "facebook/opt-iml-1.3b": [lmppl.LM, 8],  # 1.3B
}


def get_ppl(scoring_model, data, data_name, data_split, batch_size):
    # dataset setup
    encoder_decoder = type(scoring_model) is lmppl.EncoderDecoderLM
    dataset = load_dataset(data, data_name, split=data_split)
    if data_name == "Quadruples_Green_set":
        dataset_prompt = [prompt([ast.literal_eval(i['stem']) + c for c in i['pairs']], encoder_decoder) for i in dataset]
    elif data_name in ["Pairs_Cardillo_set", "Pairs_Jankowiac_set"]:
        dataset_prompt = [prompt(i['sentences'], encoder_decoder, is_sentence=True) for i in dataset]
    else:
        raise ValueError(f"unknown dataset {data_name}")

    # prompt data
    dataset_index, dataset_flat = [], []
    for n, i in enumerate(dataset_prompt):
        dataset_flat += i
        dataset_index += [n] * len(i)

    # get scores
    scores = {"answer": dataset['answer'], "labels": dataset['labels']}
    for _i, _type in zip([0, 1, 2], ["metaphor", "literal", "anomaly"]):
        _dataset_flat = [i[_i] for i in dataset_flat]
        if encoder_decoder:
            ppls = scoring_model.get_perplexity(input_texts=[x[0] for x in _dataset_flat], output_texts=[x[1] for x in _dataset_flat], batch=batch_size)
            scores[_type] = [{"input": x[0], "output": x[1], "score": float(p), "index": ind} for x, p, ind in zip(_dataset_flat, ppls, dataset_index)]
        else:
            ppls = scoring_model.get_perplexity(input_texts=_dataset_flat, batch=batch_size)
            scores[_type] = [{"input": x, "output": "", "score": float(p), "index": ind} for x, p, ind in zip(_dataset_flat, ppls, dataset_index)]
    return scores


if __name__ == '__main__':
    os.makedirs('metaphor_results/scores_instruction', exist_ok=True)

    # compute perplexity
    for target_model in language_models.keys():
        scorer = None
        lm_class, batch = language_models[target_model]
        for target_data, target_data_name, target_split in dataset_list:

            scores_file = f"metaphor_results/scores_instruction/{os.path.basename(target_model)}.{os.path.basename(target_data)}_{target_data_name}_{target_split}.json"
            if not os.path.exists(scores_file):
                if scorer is None:
                    scorer = lm_class(target_model, max_length=256) if lm_class is lmppl.MaskedLM else lm_class(target_model, device_map='auto', low_cpu_mem_usage=True)
                logging.info(f"[COMPUTING PERPLEXITY] model: `{target_model}`, data: `{target_data}/{target_data_name}/{target_split}`")
                scores_dict = get_ppl(scorer, target_data, target_data_name, target_split, batch)
                with open(scores_file, 'w') as f:
                    json.dump(scores_dict, f)

        del scorer
        gc.collect()
        torch.cuda.empty_cache()

