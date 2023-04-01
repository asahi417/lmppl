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

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
label_template = {"metaphor": "is a metaphor.", "literal": "is literal.", "anomaly": "is difficult to interpret."}


def template_four_words(four_words: List, separate_in_out: bool):
    assert len(four_words) == 4, len(four_words)
    if separate_in_out:
        return [f'{four_words[0]} is to {four_words[1]}', f'{four_words[2]} is to {four_words[3]}']
    return f'{four_words[0]} is to {four_words[1]} what {four_words[2]} is to {four_words[3]}'


def template_sentence(sentence: str, separate_in_out: bool):
    if separate_in_out:
        for phrase in [' is ', ' was ', ' were ', ' are ']:
            if phrase in sentence:
                a, b = sentence.split(phrase)
                return [f"{a} {phrase}", b]
        else:
            print("WARNING")
            input(sentence)
    return sentence


dataset_list = [  # dataset, dataset_name, split
    ['Joanne/Metaphors_and_Analogies', "Quadruples_Green_set", "test"],
    ['Joanne/Metaphors_and_Analogies', 'Pairs_Cardillo_set', "test"],
    ['Joanne/Metaphors_and_Analogies', 'Pairs_Jankowiac_set', "test"],
    # ["Joanne/katz1980_set_A", None, "test"]
]

language_models = {
    "gpt2": [lmppl.LM, 2],  # 124M
    # "facebook/opt-66b": [lmppl.LM, 1],  # 66B
    "facebook/galactica-30b": [lmppl.LM, 1],  # 30B
    "facebook/galactica-6.7b": [lmppl.LM, 1],  # 6.7B
    "facebook/galactica-1.3b": [lmppl.LM, 1],  # 1.3B
    "facebook/galactica-125m": [lmppl.LM, 1],  # 125
    "google/ul2": [lmppl.EncoderDecoderLM, 1],  # 20B
    "google/flan-ul2": [lmppl.EncoderDecoderLM, 1],  # 20B
    "EleutherAI/gpt-neox-20b": [lmppl.LM, 1],  # 20B
    "facebook/opt-iml-30b": [lmppl.LM, 1],  # 30B
    "facebook/opt-iml-max-30b": [lmppl.LM, 1],  # 30B
    "facebook/opt-30b": [lmppl.LM, 1],  # 30B
    "google/flan-t5-xxl": [lmppl.EncoderDecoderLM, 1],  # 11B
    "t5-11b": [lmppl.EncoderDecoderLM, 1],  # 11B
    "t5-3b": [lmppl.EncoderDecoderLM, 4],  # 3B
    "EleutherAI/gpt-j-6B": [lmppl.LM, 4],  # 6B
    "google/flan-t5-xl": [lmppl.EncoderDecoderLM, 4],  # 3B
    "EleutherAI/gpt-neo-2.7B": [lmppl.LM, 8],  # 2.7B
    "gpt2-xl": [lmppl.LM, 8],  # 1.5B
    "EleutherAI/gpt-neo-1.3B": [lmppl.LM, 8],  # 1.3B
    "facebook/opt-iml-max-1.3b": [lmppl.LM, 8],  # 1.3B
    "facebook/opt-iml-1.3b": [lmppl.LM, 8],  # 1.3B
    "facebook/opt-1.3b": [lmppl.LM, 8],  # 1.3B
    "google/flan-t5-large": [lmppl.EncoderDecoderLM, 256],  # 770M
    "google/flan-t5-base": [lmppl.EncoderDecoderLM, 1024],  # 220M
    "google/flan-t5-small": [lmppl.EncoderDecoderLM, 1024],  # 60M
    "t5-small": [lmppl.EncoderDecoderLM, 512],  # 60M
    "facebook/opt-350m": [lmppl.LM, 128],  # 350M
    "facebook/opt-125m": [lmppl.LM, 256],  # 125M
    "t5-large": [lmppl.EncoderDecoderLM, 128],  # 770M
    "t5-base": [lmppl.EncoderDecoderLM, 512],  # 220M
    "EleutherAI/gpt-neo-125M": [lmppl.LM, 256],  # 125M
    "gpt2-large": [lmppl.LM, 128],  # 774M
    "gpt2-medium": [lmppl.LM, 256],  # 355M
    "bert-large-cased": [lmppl.MaskedLM, 256],  # 355M
    "bert-base-cased": [lmppl.MaskedLM, 256],  # 110M
    "roberta-large": [lmppl.MaskedLM, 256],  # 355M
    "roberta-base": [lmppl.MaskedLM, 256],  # 110M
}


def get_ppl(scoring_model, data, data_name, data_split, batch_size):
    # dataset setup
    encoder_decoder = type(scoring_model) is lmppl.EncoderDecoderLM
    dataset = load_dataset(data, data_name, split=data_split)
    if data_name == "Quadruples_Green_set":
        dataset_prompt = [[template_four_words(ast.literal_eval(i['stem']) + c, encoder_decoder) for c in i['pairs']] for i in dataset]
    elif data_name in ["Pairs_Cardillo_set", "Pairs_Jankowiac_set"]:
        dataset_prompt = [[template_sentence(s, encoder_decoder) for s in i['sentences']] for i in dataset]
    else:
        raise ValueError(f"unknown dataset {data_name}")

    # prompt data
    dataset_index, dataset_flat = [], []
    for n, i in enumerate(dataset_prompt):
        dataset_flat += i
        dataset_index += [n] * len(i)

    # get scores
    scores = {"answer": dataset['answer'], "labels": dataset['labels']}
    if encoder_decoder:
        ppls = scoring_model.get_perplexity(input_texts=[x[0] for x in dataset_flat], output_texts=[x[1] for x in dataset_flat], batch=batch_size)
        scores["perplexity"] = [{"input": x[0], "output": x[1], "score": float(p), "index": ind} for x, p, ind in zip(dataset_flat, ppls, dataset_index)]
    else:
        print(dataset_flat[:2], len(dataset_flat))
        ppls = scoring_model.get_perplexity(input_texts=dataset_flat[:2])
        ppls = scoring_model.get_perplexity(input_texts=dataset_flat[:10], batch=batch_size)
        ppls = scoring_model.get_perplexity(input_texts=dataset_flat, batch=batch_size)
        scores["perplexity"] = [{"input": x, "output": "", "score": float(p), "index": ind} for x, p, ind in zip(dataset_flat, ppls, dataset_index)]
    return scores


if __name__ == '__main__':
    os.makedirs('metaphor_results/scores_no_prompt', exist_ok=True)

    # compute perplexity
    for target_model in language_models.keys():
        scorer = None
        lm_class, batch = language_models[target_model]
        for target_data, target_data_name, target_split in dataset_list:
            scores_file = f"metaphor_results/scores_no_prompt/{os.path.basename(target_model)}.{os.path.basename(target_data)}_{target_data_name}_{target_split}.json"
            if not os.path.exists(scores_file):
                if scorer is None:
                    # scorer = lm_class(target_model, max_length=256) if lm_class is lmppl.MaskedLM else lm_class(target_model, device_map='auto', low_cpu_mem_usage=True)
                    scorer = lm_class(target_model, max_length=256) if lm_class is lmppl.MaskedLM else lm_class(target_model)
                logging.info(f"[COMPUTING PERPLEXITY] model: `{target_model}`, data: `{target_data}/{target_data_name}/{target_split}`")
                scores_dict = get_ppl(scorer, target_data, target_data_name, target_split, batch)
                with open(scores_file, 'w') as f:
                    json.dump(scores_dict, f)

        del scorer
        gc.collect()
        torch.cuda.empty_cache()

