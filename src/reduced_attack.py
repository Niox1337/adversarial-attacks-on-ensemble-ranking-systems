import os
import time
import logging
import argparse
import copy
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from utils import read_qds_pairs
from modeling import RankingBERT_Train
from marcodoc.dataset import CollectionDataset
from myutils.semantic_helper import SemanticHelper
from myutils.word_recover.Bert_word_recover import BERTWordRecover
from myutils.attacker.attacker import Attacker
import ir_datasets

logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
    datefmt="%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/train")
    parser.add_argument("--msmarco_dir", type=str, default="./data/msmarco-passage")
    parser.add_argument(
        "--collection_memmap_dir", type=str, default="./data/collection_memmap"
    )
    parser.add_argument("--tokenize_dir", type=str, default="./data/tokenize")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--bert_tokenizer_path", type=str, default="bert-base-uncased")
    parser.add_argument("--eps", type=float, default=45)
    parser.add_argument("--max_iter", type=int, default=3)
    parser.add_argument("--simi_threshod", type=float, default=0.7)
    parser.add_argument("--max_attack_word_number", type=int, default=10)
    parser.add_argument("--ori_model_path", type=str, default="bert-base-uncased")
    parser.add_argument("--surrogate_model_path", type=str, default="bert-base-uncased")
    parser.add_argument("--embed_path", type=str, default="./glove.6B.50d.txt")
    parser.add_argument(
        "--embed_cos_matrix_path", type=str, default="./cosine_matrix.npy"
    )
    return parser.parse_args()


def initialize_models(args):
    """Initialize original and surrogate models."""

    def load_model(model_path):
        config = BertConfig.from_pretrained(model_path)
        model = RankingBERT_Train.from_pretrained(model_path, config=config)
        return model.to(args.device)

    ori_model = load_model(args.ori_model_path)
    surr_model = load_model(args.surrogate_model_path)
    if args.n_gpu > 1:
        ori_model = torch.nn.DataParallel(ori_model)
        surr_model = torch.nn.DataParallel(surr_model)

    return ori_model, surr_model


def prepare_batch(bert_tokenizer, query, doc_texts, max_length):
    """Prepare a batch of input tensors for the model."""
    inputs = [
        bert_tokenizer(
            query,
            doc,
            max_length=max_length,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt",
        )
        for doc in doc_texts
    ]
    batch = {k: torch.cat([inp[k] for inp in inputs], dim=0) for k in inputs[0]}
    return batch


def perform_attack(
    ori_model,
    surr_model,
    batch,
    doc_ids,
    attack_doc_id,
    query_id,
    args,
    word_re,
    semantic_helper,
    ori_score,
    ori_qds,
):
    """Perform adversarial attack on the top-ranked document."""
    attacker = Attacker()
    attack_doc_input, batch = find_and_remove_doc(batch, doc_ids, attack_doc_id)

    attacker.get_model_gradient(surr_model, batch, attack_doc_input, args.device)
    topk_words = word_re.get_highest_gradient_words(
        surr_model, args.max_attack_word_number
    )

    attacker.attack(surr_model, batch, attack_doc_input, args.eps, args.max_iter)
    sim_word_ids = semantic_helper.pick_most_similar_words_batch(topk_words)

    return word_re.recover_document(
        attack_doc_id, sim_word_ids, ori_score, ori_model, query_id, args
    )


def main():
    args = parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    dataset = ir_datasets.load("antique/test")
    ori_model, surr_model = initialize_models(args)
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)

    synonym_helper = SemanticHelper(args.embed_path, args.embed_cos_matrix_path)
    synonym_helper.build_vocab()
    word_recover = BERTWordRecover(args.bert_tokenizer_path)

    for query in dataset.queries_iter():
        query_text = query.text
        relevant_docs = [
            doc.doc_id for doc in dataset.qrels_iter() if doc.query_id == query.query_id
        ]
        doc_texts = [
            doc.text for doc in dataset.docs_iter() if doc.doc_id in relevant_docs
        ]

        batch = prepare_batch(
            bert_tokenizer,
            query_text,
            doc_texts,
            args.max_query_length + args.max_doc_length,
        )
        with torch.no_grad():
            scores = ori_model(batch).cpu().numpy().tolist()

        rankings = sorted(zip(relevant_docs, scores), key=lambda x: x[1], reverse=True)
        top_doc_id, top_score = rankings[0]

        logger.info("Top-ranked doc: %s with score %.4f", top_doc_id, top_score)

        new_doc_tokens = perform_attack(
            ori_model,
            surr_model,
            batch,
            relevant_docs,
            top_doc_id,
            query.query_id,
            args,
            word_recover,
            synonym_helper,
            top_score,
            None,
        )

        logger.info(
            "Adversarial attack successful! Modified doc tokens: %s", new_doc_tokens
        )


if __name__ == "__main__":
    main()
