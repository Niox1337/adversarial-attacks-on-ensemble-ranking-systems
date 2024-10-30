import os
import math
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from collections import namedtuple, defaultdict
from transformers import BertTokenizer
from torch.utils.data import Dataset

import random
from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Optional, Union
import ir_datasets as irds
from rankers_util import initialise_irds_eval
from rankers.datasets.corpus import Corpus

logger = logging.getLogger(__name__)



class PreAttackDataset(Dataset):
    def __init__(self, 
                 data : pd.DataFrame, 
                 corpus : Union[Corpus, irds.Dataset],
                 ) -> None:
        super().__init__()
        self.training_data = data
        self.corpus = corpus
        self.__post_init__()

    
    def __post_init__(self):
        for column in 'query_id', 'doc_id_a':
            if column not in self.training_data.columns: raise ValueError(f"Format not recognised, Column '{column}' not found in triples dataframe")
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()
        self.training_data = [*self.training_data.itertuples(index=False)]

    @classmethod
    def from_irds(cls,
                    ir_dataset : str,
                    collate_fn : Optional[callable] = lambda x : pd.DataFrame(x.docpairs_iter()) 
                    ) -> 'PreAttackDataset':
            dataset = irds.load(ir_dataset)
            assert dataset.has_docpairs(), "Dataset does not have docpairs, check you are not using a test collection"
            training_data = collate_fn(dataset)
            return cls(training_data, dataset)
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        item = self.training_data[idx]
        qid, doc_id_a, = item.query_id, item.doc_id_a
        query = self.queries[str(qid)]
        text = self.docs[str(doc_id_a)]

        return qid, query, doc_id_a, text
    
class PostAttackDataset(Dataset):
    def __init__(self, 
                 data : pd.DataFrame, 
                 corpus : Union[Corpus, irds.Dataset],
                 ) -> None:
        super().__init__()
        self.training_data = data
        self.corpus = corpus
        self.__post_init__()

    
    def __post_init__(self):
        for column in 'query_id', 'doc_id_a', 'perturbed':
            if column not in self.training_data.columns: raise ValueError(f"Format not recognised, Column '{column}' not found in triples dataframe")
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()
        self.training_data = [*self.training_data.itertuples(index=False)]

    @classmethod
    def from_irds(cls,
                    ir_dataset : str,
                    collate_fn : Optional[callable] = lambda x : pd.DataFrame(x.docpairs_iter()) 
                    ) -> 'PreAttackDataset':
            dataset = irds.load(ir_dataset)
            assert dataset.has_docpairs(), "Dataset does not have docpairs, check you are not using a test collection"
            training_data = collate_fn(dataset)
            return cls(training_data, dataset)
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        item = self.training_data[idx]
        qid, doc_id_a, perturbed = item.query_id, item.doc_id_a, perturbed
        query = self.queries[str(qid)]
        text = self.docs[str(doc_id_a)]

        return qid, query, doc_id_a, text, perturbed

class PreCatDataCollator:
    def __init__(self, 
                 tokenizer,
                 q_max_length=30,
                 d_max_length=200,
                 ) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch) -> dict:
        qidx, queries, didx, docs = zip(*batch)

        tokenized_sequences = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation='only_second',
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )
        return {
            "sequences": dict(tokenized_sequences),
            "qidx": torch.tensor(qidx),
            "didx": torch.tensor(didx),
        }
    
class PostCatDataCollator:
    def __init__(self, 
                 tokenizer,
                 q_max_length=30,
                 d_max_length=200,
                 ) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch) -> dict:
        qidx, queries, didx, docs, perturbed = zip(*batch)

        tokenized_sequences = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation='only_second',
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )

        tokenized_perturbed_sequences = self.tokenizer(
            queries,
            perturbed,
            padding=True,
            truncation='only_second',
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )

        return {
            "sequences": dict(tokenized_sequences),
            "perturbed_sequences": dict(tokenized_perturbed_sequences),
            "qidx": torch.tensor(qidx),
            "didx": torch.tensor(didx),
        }

