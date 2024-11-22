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
from rankers._util import initialise_irds_eval
from rankers.datasets.corpus import Corpus

logger = logging.getLogger(__name__)


class PreAttackDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        corpus: Union[Corpus, irds.Dataset],
        teacher_data: dict = None,
    ) -> None:
        super().__init__()
        self.training_data = data
        self.corpus = corpus
        self.teacher_data = teacher_data
        self.__post_init__()

    def __post_init__(self):
        for column in ["query_id", "doc_id_a", "doc_id_b"]:
            if column not in self.training_data.columns:
                raise ValueError(
                    f"Format not recognised, Column '{column}' not found in triples dataframe"
                )
        self.docs = (
            pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        )
        self.queries = (
            pd.DataFrame(self.corpus.queries_iter())
            .set_index("query_id")["text"]
            .to_dict()
        )
        self.training_data = [*self.training_data.itertuples(index=False)]

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        item = self.training_data[idx]
        qid, doc_id_a, doc_id_b = item.query_id, item.doc_id_a, item.doc_id_b
        query = self.queries[str(qid)]
        a_texts = (
            [self.docs[str(doc_id_a)]]
            if isinstance(doc_id_a, str)
            else [self.docs[str(doc_id)] for doc_id in doc_id_a]
        )

        if isinstance(doc_id_b, list):
            b_texts = [self.docs[str(doc_id)] for doc_id in doc_id_b]
        else:
            b_texts = [self.docs[str(doc_id_b)]]
            doc_id_b = [doc_id_b]

        if self.teacher_data is not None:
            if qid not in self.teacher_data:
                raise ValueError(f"Query ID {qid} not found in teacher data")
            if doc_id_a not in self.teacher_data[qid]:
                raise ValueError(
                    f"Doc ID {doc_id_a} not found in teacher data for query {qid}"
                )
            if any(doc_id not in self.teacher_data[qid] for doc_id in doc_id_b):
                raise ValueError(f"Doc ID not found in teacher data for query {qid}")
            doc_id_a_score = (
                [self.teacher_data[qid][doc_id_a]]
                if isinstance(doc_id_a, str)
                else [self.teacher_data[qid][doc_id] for doc_id in doc_id_a]
            )
            doc_id_b_scores = [self.teacher_data[qid][doc_id] for doc_id in doc_id_b]
        else:
            doc_id_a_score = []
            doc_id_b_scores = []

        return (
            qid,
            query,
            doc_id_a,
            doc_id_b,
            a_texts,
            b_texts,
            doc_id_a_score,
            doc_id_b_scores,
        )


class PostAttackDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        corpus: Union[Corpus, irds.Dataset],
    ) -> None:
        super().__init__()
        self.training_data = data
        self.corpus = corpus
        self.__post_init__()

    def __post_init__(self):
        for column in "query_id", "doc_id_a", "perturbed":
            if column not in self.training_data.columns:
                raise ValueError(
                    f"Format not recognised, Column '{column}' not found in triples dataframe"
                )
        self.docs = (
            pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        )
        self.queries = (
            pd.DataFrame(self.corpus.queries_iter())
            .set_index("query_id")["text"]
            .to_dict()
        )
        self.training_data = [*self.training_data.itertuples(index=False)]

    @classmethod
    def from_irds(
        cls,
        ir_dataset: str,
        collate_fn: Optional[callable] = lambda x: pd.DataFrame(x.docpairs_iter()),
    ) -> "PreAttackDataset":
        dataset = irds.load(ir_dataset)
        assert (
            dataset.has_docpairs()
        ), "Dataset does not have docpairs, check you are not using a test collection"
        training_data = collate_fn(dataset)
        return cls(training_data, dataset)

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        item = self.training_data[idx]
        qid, doc_id_a, perturbed = item.query_id, item.doc_id_a, item.perturbed
        query = self.queries[str(qid)]
        text = self.docs[str(doc_id_a)]

        return qid, query, doc_id_a, text, perturbed


class PreCatDataCollator:
    def __init__(
        self,
        tokenizer,
        q_max_length=30,
        d_max_length=200,
    ) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch) -> dict:
        batch_qids = []
        batch_queries_a = []
        batch_queries_b = []
        batch_a_docs = []
        batch_b_docs = []
        batch_id_a = []
        batch_id_b = []
        scores_a = []
        scores_b = []

        for item in batch:
            (
                qid,
                query,
                doc_id_a,
                doc_id_b,
                a_texts,
                b_texts,
                doc_id_a_score,
                doc_id_b_scores,
            ) = item
            batch_qids.append(qid)
            batch_queries_a.extend([query] * len(a_texts))
            batch_queries_b.extend([query] * len(b_texts))
            batch_a_docs.extend(a_texts)
            batch_b_docs.extend(b_texts)
            batch_id_a.extend(doc_id_a)
            batch_id_b.extend(doc_id_b)
            scores_a.extend(doc_id_a_score)
            scores_b.extend(doc_id_b_scores)

        tokenized_sequences_a = self.tokenizer(
            batch_queries_a,
            batch_a_docs,
            padding=True,
            truncation="only_second",
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )
        tokenized_sequences_b = self.tokenizer(
            batch_queries_b,
            batch_b_docs,
            padding=True,
            truncation="only_second",
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )
        return {
            "sequences_a": dict(tokenized_sequences_a),
            "sequences_b": dict(tokenized_sequences_b),
            "qid": torch.tensor(batch_qids),
            "docid_a": torch.tensor(batch_id_a),
            "docid_b": torch.tensor(batch_id_b),
            "scores_a": torch.tensor(scores_a) if len(scores_a) > 0 else None,
            "scores_b": torch.tensor(scores_b) if len(scores_b) > 0 else None,
        }


class PreDotDataCollator:
    def __init__(
        self,
        tokenizer,
        q_max_length=30,
        d_max_length=200,
    ) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch) -> dict:
        qidx, queries, didx, docs = zip(*batch)

        tokenized_queries = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.q_max_length,
            return_tensors="pt",
        )
        tokenized_docs = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="pt",
        )

        return {
            "queries": dict(tokenized_queries),
            "docs": dict(tokenized_docs),
            "qid": torch.tensor(qidx),
            "docid": torch.tensor(didx),
        }


class PostCatDataCollator:
    def __init__(
        self,
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
            truncation="only_second",
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )

        tokenized_perturbed_sequences = self.tokenizer(
            queries,
            perturbed,
            padding=True,
            truncation="only_second",
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )

        return {
            "sequences": dict(tokenized_sequences),
            "perturbed_sequences": dict(tokenized_perturbed_sequences),
            "qid": torch.tensor(qidx),
            "docid": torch.tensor(didx),
        }


class PostDotDataCollator:
    def __init__(
        self,
        tokenizer,
        q_max_length=30,
        d_max_length=200,
    ) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch) -> dict:
        qidx, queries, didx, docs, perturbed = zip(*batch)

        tokenized_queries = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.q_max_length,
            return_tensors="pt",
        )
        tokenized_docs = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="pt",
        )
        tokenized_perturbed_docs = self.tokenizer(
            perturbed,
            padding=True,
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="pt",
        )

        return {
            "queries": dict(tokenized_queries),
            "docs": dict(tokenized_docs),
            "perturbed_docs": dict(tokenized_perturbed_docs),
            "qid": torch.tensor(qidx),
            "docid": torch.tensor(didx),
        }
