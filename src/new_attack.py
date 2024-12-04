from torch.utils.data import DataLoader
import ir_datasets as irds
from hybridattack.myutils.modelling.modelling_bert_cat import Cat
from hybridattack.myutils.modelling.modelling_bert_dot import Dot
from hybridattack.myutils.semantic_helper import SemanticHelper
from hybridattack.marcopassage.dataset import (
    PreAttackDataset,
    PreCatDataCollator,
    PreDotDataCollator,
)
from hybridattack.myutils.word_recover.Bert_word_recover import BERTWordRecover
import json
from fire import Fire
import torch
import copy
from hybridattack.myutils.attacker.attacker import Attacker
from transformers import HfArgumentParser
import pandas as pd
from dataclasses import dataclass, field
from src import load_json


@dataclass
class Args:
    model_name_or_path: str = field(default="crystina-z/monoELECTRA_LCE_nneg31")
    output_dir: str = field(default="./output")
    dataset: str = field(default="./dataset.tsv.gz")
    scores: str = field(default="./scores.json.gz")
    model_type: str = field(default="cat")
    subset: int = field(default=1)
    max_attack_word_number: int = field(default=10)
    toy: False = field(default=False)
    eps: float = field(default=0.1)
    max_iter: int = field(default=3)
    simi_candi_topk: int = field(default=5)
    simi_threshod: float = field(default=0.7)
    max_doc_length: int = field(default=200)
    max_query_length: int = field(default=30)
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")
    find_word_search: bool = field(default=False)
    doc_list_length: int = field(default=100)
    dataset_id : str = field(default="msmarco-passage/trec-dl-2019/judged")
    group_size: int = field(default=20)

    def __post_init(self):
        if self.model_type not in ["cat", "dot"]:
            raise ValueError(f"Model type '{self.model_type}' not recognised")
        self.dataset_frame = pd.read_csv(self.dataset, sep="\t")
        self.scores_lookup = load_json(self.scores)


def load_local_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    # Ensure the data is structured as a list of texts
    return [{"text": entry["text"]} for entry in data]


# Make new semantic helper without pre-compute
def replace_with_similar_words(
    input_text: str,
    semantic_helper: SemanticHelper,
    word_re: BERTWordRecover,
    top_k: int = 5,
    threshold: float = 0.7,
):
    tokens = semantic_helper.tokenizer.tokenize(input_text)
    token_ids = semantic_helper.tokenizer.convert_tokens_to_ids(tokens)
    similar_words, *_ = semantic_helper.pick_most_similar_words_batch(
        tokens,
        ret_count=top_k,
        threshold=threshold,
        word_re=word_re,
        ori_doc_token_ids_list=token_ids,
    )
    new_tokens = []
    for token in tokens:
        if token in similar_words and similar_words[token]:
            new_token = similar_words[token][
                len(similar_words[token]) // 2
            ]  # Take the medium similar word
        else:
            new_token = token  # Keep the original word if no similar word is found
        new_tokens.append(new_token)

    adversarial_text = semantic_helper.tokenizer.convert_tokens_to_string(new_tokens)
    return adversarial_text


def attack(
    ranking,
    sequence_score,
    qid,
    attack_doc_id,
    model,
    word_re,
    semantic_helper,
    tokenizer,
    args,
    device,
    attack_word_number=5,
):
    attack_sequence = ranking["input_ids"][0]
    batch_list = ranking["input_ids"][1:]
    pre_sep = attack_sequence[1 : attack_sequence.index(tokenizer.sep_token_id)]
    post_sep = attack_sequence[attack_sequence.index(tokenizer.sep_token_id) + 1 :]
    post_sep = post_sep[:-2]
    attack_sequence_list = list({}.fromkeys(post_sep).keys())

    word_embedding_matrix = word_re.get_word_embedding(model)
    ori_we_matrix = word_embedding_matrix.clone().detach()

    attacker = Attacker()

    attacker.get_model_gradient(model, batch_list, attack_sequence, device)

    gradient_norm_topk_word, gradient_topk_word_idx_list = (
        word_re.get_highest_gradient_words(
            model, attack_word_number, attack_sequence_list
        )
    )

    gradient_topk_words = []
    for word_idx in gradient_topk_word_idx_list:
        gradient_topk_words.append(word_re.idx2word[word_idx])

    attacker.attack(
        model,
        batch_list,
        attack_sequence,
        attack_word_idx=attack_sequence_list,
        args=args,
        eps=args.eps,
        max_iter=args.max_iter,
    )

    attacked_we_matrix = word_re.get_word_embedding(model)

    sim_word_ids_dict, sim_values, sub_word_dict = (
        semantic_helper.pick_most_similar_words_batch(
            gradient_topk_words,
            attack_sequence_list,
            word_re,
            args.simi_candi_topk,
            args.simi_threshod,
        )
    )

    new_doc_token_id_list, score = word_re.recover_document_greedy_rank_pos(
        post_sep,
        ori_we_matrix,
        attacked_we_matrix,
        sim_word_ids_dict,
        sequence_score,
        model,
        pre_sep,
        args,
        sub_word_dict,
        args.score_lookup,
        attack_doc_id,
        qid,
    )

    return new_doc_token_id_list, score


def main():
    parser = HfArgumentParser(Args)
    args = parser.parse_args()[0]

    if args.model_type == "cat":
        model = Cat.from_pretrained(args.model_name_or_path)
    elif args.model_type == "dot":
        model = Dot.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Model type '{args.model_type}' not recognised")

    model_state_dict = copy.deepcopy(model.state_dict())

    tokenizer = model.tokenizer
    embeddings = model.model.base_model.embeddings.word_embeddings
    semantic_helper = SemanticHelper(embeddings, tokenizer)
    corpus = irds.load(args.dataset_id)
    word_re = BERTWordRecover(
        tokenizer,
        max_query_length=args.max_query_length,
        max_doc_length=args.max_doc_length,
        corpus=corpus
    )

    if not args.toy:
        dataset = PreAttackDataset(args.dataset_data, corpus, args.score_lookup)
        collator = (
            PreCatDataCollator(tokenizer)
            if args.model_type == "cat"
            else PreDotDataCollator(tokenizer)
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator)
    else:
        dataloader = [
            {
                'qid' : "0",
                'docid_a' : ["10"],
                'sequences_a' : tokenizer(['']),
                'sequences_b' : tokenizer(['']),
            }
        ]
    for batch in dataloader:
        attack_text_dict = {}
        attack_score_dict = {}
        for i in range(len(batch["sequences_a"])):
            sequence_a = batch["sequences_a"][i]
            doc_id_a = batch["docid_a"][i]
            qid = batch["qid"]

            score_lookup = {
                doc_id: score
                for doc_id, score in zip(batch["docid_a"], batch["scores_a"])
            }
            score_lookup.update(
                {
                    doc_id: score
                    for doc_id, score in zip(batch["docid_b"], batch["scores_b"])
                }
            )

            ranking = torch.cat(
                [sequence_a, batch["sequences_b"]], dim=0
            )  # Concatenate the two sequences

            new_tokens, score = attack(
                # ranking,
                # score_lookup,
                # model,
                # word_re,
                # semantic_helper,
                # tokenizer,
                # args,
                # args.device,
                ranking,
                score_lookup,
                qid,
                doc_id_a,
                model,
                word_re,
                semantic_helper,
                tokenizer,
                args,
                args.device,
                attack_word_number=5,
            )

            new_key = f"{qid}_{doc_id_a}"
            attack_text_dict[new_key] = new_tokens
            attack_score_dict[new_key] = score

        for key, text in attack_text_dict.items():
            recovered_doc = word_re.recover_doc(
                key.split("_")[1], text, args.max_doc_length
            )
            recovered_attack_doc = word_re.tokenizer.convert_tokens_to_string(
                text
            )

            to_write = f"{key}\t{recovered_doc}\t{recovered_attack_doc}\t{attack_score_dict[key]}"

            if args.toy:
                print(to_write)
            else:
                with open(args.output_file, "a") as f:
                    f.write(to_write)


if __name__ == "__main__":
    main()

    # word recoverer
    # new attack function which expects our form
