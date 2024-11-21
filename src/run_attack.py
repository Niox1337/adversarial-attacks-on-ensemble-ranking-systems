# import copy
# import logging, argparse, os, time
# import ir_datasets
# from tqdm import tqdm
# import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from rankers import Cat, Dot
from hybridattack.myutils.semantic_helper import SemanticHelper
from hybridattack.marcopassage.dataset import PreAttackDataset, PreCatDataCollator
from hybridattack.myutils.word_recover.Bert_word_recover import BERTWordRecover
import json

def load_local_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Ensure the data is structured as a list of texts
    return [{"text": entry["text"]} for entry in data]

# Make new semantic helper without pre-compute
def replace_with_similar_words(input_text: str, semantic_helper: SemanticHelper, word_re: BERTWordRecover, top_k: int=5, threshold: float=0.7):
    tokens = semantic_helper.tokenizer.tokenize(input_text)
    token_ids = semantic_helper.tokenizer.convert_tokens_to_ids(tokens)
    similar_words, *_ = semantic_helper.pick_most_similar_words_batch(tokens, ret_count=top_k, threshold=threshold, word_re=word_re, ori_doc_token_ids_list=token_ids)
    new_tokens = []
    for token in tokens:
        if token in similar_words and similar_words[token]:
            new_token = similar_words[token][len(similar_words[token])//2]  # Take the medium similar word
        else:
            new_token = token  # Keep the original word if no similar word is found
        new_tokens.append(new_token)

    adversarial_text = semantic_helper.tokenizer.convert_tokens_to_string(new_tokens)
    return adversarial_text


def main(model_name_or_path : str, output_dir, dataset : str, model_type : str = 'cat', subset : int = 1):

    if model_type == 'cat': model = Cat.from_pretrained(model_name_or_path)
    elif model_type == 'dot': model = Dot.from_pretrained(model_name_or_path)
    else: raise ValueError(f"Model type '{model_type}' not recognised")

    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    embeddings = model.model.base_model.embeddings.word_embeddings
    semantic_helper = SemanticHelper(embeddings, tokenizer)
    word_re = BERTWordRecover(tokenizer, max_query_length=64, max_doc_length=512)

    # dataset = PreAttackDataset.from_irds(dataset)
    dataset = load_local_json(dataset)
    collator = PreCatDataCollator(tokenizer)

    # dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator)

    for batch in dataset:
        input_text = batch["text"]
        adversarial_text = replace_with_similar_words(
            input_text=input_text,
            semantic_helper=semantic_helper,
            word_re=word_re,
            top_k=5,
            threshold=0.7,
        )
        print("Original:", input_text)
        print("Adversarial:", adversarial_text)

if __name__ == "__main__":
    main(
        model_name_or_path="bert-base-uncased",
        output_dir="./output",
        dataset="./dataset.json",
        model_type="cat",
        subset=1
    )

    # word recoverer 
    # new attack function which expects our form 

    




