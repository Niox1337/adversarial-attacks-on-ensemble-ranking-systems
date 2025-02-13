from torch.utils.data import DataLoader
import ir_datasets as irds
from src.hybridattack.myutils.modelling.modelling_bert_cat import Cat
from src.hybridattack.myutils.modelling.modelling_bert_dot import Dot
from src.hybridattack.myutils.semantic_helper import SemanticHelper
from src.hybridattack.marcopassage.dataset import (
    PreAttackDataset,
    PreCatDataCollator,
    PreDotDataCollator,
)
from src.hybridattack.myutils.word_recover.Bert_word_recover import BERTWordRecover
import json
import torch
import copy
from src.hybridattack.myutils.attacker.attacker import Attacker
from transformers import HfArgumentParser
import pandas as pd
from dataclasses import dataclass, field
from src import load_json
from src.get_topk import get_topk


@dataclass
class Args:
    model_name_or_path: str = field(default="crystina-z/monoELECTRA_LCE_nneg31")
    output_file: str = field(default="./output/attacks")
    dataset: str = field(default="./dataset.tsv.gz")
    scores: str = field(default="./scores.json.gz")
    model_type: str = field(default="cat")
    subset: int = field(default=1)
    max_attack_word_number: int = field(default=10)
    toy: bool = field(default=False)
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
    attack_sequence = ranking["input_ids"][0] # SEQ_LEN X 1
    sep_idx = attack_sequence.tolist().index(tokenizer.sep_token_id)
    pre_sep = attack_sequence[1 : sep_idx]
    post_sep = attack_sequence[sep_idx + 1 :]
    post_sep = post_sep[post_sep > 102]
    post_sep = post_sep[:-2]
    attack_sequence_list = [
        token_id for token_id in dict.fromkeys(post_sep).keys()
    ]
    word_embedding_matrix = word_re.get_word_embedding(model)
    ori_we_matrix = word_embedding_matrix.clone()

    attacker = Attacker()

    attacker.get_model_gradient(model, ranking, device)

    gradient_matrix = word_embedding_matrix.grad.clone()
    mask = torch.zeros(gradient_matrix.shape[0], dtype=torch.bool)
    mask[[token.item() for token in attack_sequence_list]] = 1
    filtered_gradient_matrix = gradient_matrix.clone()
    filtered_gradient_matrix[~mask] = 0
    gradient_norm_topk_word, gradient_topk_word_idx_list = (
        word_re.get_highest_gradient_words(
            filtered_gradient_matrix, attack_word_number
        )
    )

    gradient_topk_words = []
    for word_idx in gradient_topk_word_idx_list:
        gradient_topk_words.append(word_re.idx2word[word_idx])
    attacker.attack(
        model,
        ranking,
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
    # print('sim_word_ids_dict', sim_word_ids_dict)
    # print('sim_values', sim_values)
    # print('sub_word_dict', sub_word_dict)


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
    # we want the score to be higher
    return new_doc_token_id_list, score


def main():
    parser = HfArgumentParser(Args)
    args = parser.parse_args()
    if args.model_type == "cat":
        model = Cat.from_pretrained(args.model_name_or_path).to(args.device)
    elif args.model_type == "dot":
        model = Dot.from_pretrained(args.model_name_or_path).to(args.device)
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
        # dataset = PreAttackDataset(args.dataset_data, corpus, args.score_lookup)
        # collator = (
        #     PreCatDataCollator(tokenizer)
        #     if args.model_type == "cat"
        #     else PreDotDataCollator(tokenizer)
        # )
        # dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator)

        ranked_pairs, args.score_lookup = get_topk("./output", args.dataset_id, args.model_name_or_path)
        dataloader = []

        for qid, group in ranked_pairs.groupby("qid"):
            group = group.sort_values(by="score", ascending=False)  # Lower score = more relevant

            docid_b = group.iloc[:-1]["doc_id"].tolist()  # All except the last (most irrelevant)
            docid_a = [group.iloc[-1]["doc_id"]]  # Most irrelevant doc (highest score)

            sequences_a = tokenizer(
                [group.iloc[-1]["query"]],
                [group.iloc[-1]["text"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            sequences_b = tokenizer(
                [group.iloc[-1]["query"]] * len(docid_b),
                group.iloc[:-1]["text"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,  # Truncate if exceeding model max length
                max_length=512
            )

            scores_a = [group.iloc[-1]["score"]]  # Score for docid_a
            scores_b = group.iloc[:-1]["score"].tolist()  # Scores for docid_b

            dataloader.append({
                "qid": str(qid),
                "docid_a": docid_a,
                "docid_b": docid_b,
                "sequences_a": sequences_a,
                "sequences_b": sequences_b,
                "scores_a": scores_a,
                "scores_b": scores_b
            })


    else:
        dataloader = [
            {
                'qid' : "0",
                'docid_a' : ["1"],
                'docid_b' : ["2", "3", "4"],
                'sequences_a' : tokenizer(['sample query'], ['sample positive'], return_tensors="pt"),
                'sequences_b' : tokenizer(['sample query']*3, ['sample negative 1', 'sample negative 3', 'sample negative 3'], return_tensors="pt"),
                'scores_a': [0.1],
                'scores_b': [0.5, 0.3, 0.2]
            }
        ]

        args.score_lookup = {
            "0": {
                "1": 0.1,
                "2": 0.5,
                "3": 0.3,
                "4": 0.2
            }
        }
    for batch in dataloader:
        attack_text_dict = {}
        attack_score_dict = {}

        qid = batch["qid"]
        doc_id_a = batch["docid_a"][0]

        all_docs = [batch["sequences_a"]["input_ids"].squeeze(0)] + [
            seq.squeeze(0) for seq in batch["sequences_b"]["input_ids"]
        ]

        combined_tokens = tokenizer(
            [tokenizer.decode(seq.tolist()) for seq in all_docs],
            padding=True,
            truncation=True,
            max_length=args.max_doc_length,
            return_tensors="pt",
        )

        ranking = {
            "input_ids": combined_tokens["input_ids"],
            "token_type_ids": combined_tokens["token_type_ids"],
            "position_ids": torch.arange(combined_tokens["input_ids"].size(-1)).unsqueeze(0).repeat(
                combined_tokens["input_ids"].size(0), 1
            ),
        }

        new_tokens, score = attack(
            ranking,
            batch["scores_a"][0],
            qid,
            doc_id_a,
            model,
            word_re,
            semantic_helper,
            tokenizer,
            args,
            args.device,
            attack_word_number=args.max_attack_word_number,
        )

        new_key = f"{qid}_{doc_id_a}"
        attack_text_dict[new_key] = new_tokens
        attack_score_dict[new_key] = score

        for key, text in attack_text_dict.items():
            if isinstance(text, torch.Tensor):
                text = text.tolist()
            recovered_doc = word_re.recover_doc(
                key.split("_")[1], text, args.max_doc_length
            )
            recovered_attack_doc = word_re.tokenizer.convert_tokens_to_string(
                word_re.tokenizer.convert_ids_to_tokens(text)
            )

            to_write = f"key: {key}\nrecovered_doc: {recovered_doc}\nrecovered_attack_doc: {recovered_attack_doc}\nattack_score_dict[key]: {attack_score_dict[key]}\n"

            if args.toy:
                print(to_write)
            else:
                with open(args.output_file, "a") as f:
                    f.write(to_write)


if __name__ == "__main__":
    main()
