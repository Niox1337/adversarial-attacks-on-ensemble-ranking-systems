import copy
import logging, argparse, os, time
import ir_datasets
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from rankers import Cat, Dot
from .myutils.semantic_helper import SemanticHelper
from .msmarcopassage.dataset import PreAttackDataset, PreCatDataCollator

# Make new semantic helper without pre-compute 

def main(model_name_or_path : str, output_dir, dataset : str, model_type : str = 'cat', subset : int = 1):

    if model_type == 'cat': model = Cat.from_pretrained(model_name_or_path)
    elif model_type == 'dot': model = Dot.from_pretrained(model_name_or_path)
    else: raise ValueError(f"Model type '{model_type}' not recognised")

    semantic_helper = SemanticHelper(getattr(model, 'classifier' if model_type == 'cat' else 'encoder').embeddings)
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    dataset = PreAttackDataset.from_irds(dataset)
    collator = PreCatDataCollator(tokenizer)

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator)

    # word recoverer 
    # new attack function which expects our form 

    




