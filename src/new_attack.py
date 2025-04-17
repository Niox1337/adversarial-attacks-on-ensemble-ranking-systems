from lxml.html.diff import token
from torch.utils.data import DataLoader
import ir_datasets as irds
from src.hybridattack.myutils.modelling.modelling_bert_cat import Cat
from src.hybridattack.myutils.modelling.modelling_bert_dot import Dot
from src.hybridattack.myutils.semantic_helper import SemanticHelper
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
    output_file: str = field(default="./output/attacks.txt")
    dataset: str = field(default="./dataset.tsv.gz")
    scores: str = field(default="./scores.json.gz")
    model_type: str = field(default="cat")
    subset: int = field(default=1)
    max_attack_word_number: int = field(default=10)
    toy: bool = field(default=False)
    eps: float = field(default=0.1)
    max_iter: int = field(default=3)
    simi_candi_topk: int = field(default=5)
    simi_threshod: float = field(default=0.4)
    max_doc_length: int = field(default=300)
    max_query_length: int = field(default=30)
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")
    find_word_search: bool = field(default=False)
    doc_list_length: int = field(default=100)
    dataset_id : str = field(default="msmarco-passage/trec-dl-2019/judged")
    group_size: int = field(default=20)
    rewrite_query: bool = field(default=False)
    index_path: str = field(
        default="./index/"
    )

    def __post_init(self):
        if self.model_type not in ["cat", "dot"]:
            raise ValueError(f"Model type '{self.model_type}' not recognised")
        self.dataset_frame = pd.read_csv(self.dataset, sep="\t")
        self.scores_lookup = load_json(self.scores)


def load_local_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return [{"text": entry["text"]} for entry in data]


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
    post_sep = post_sep[post_sep > 103] # Filter special tokens
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

    """
    Uncomment the following to write the candidate substitutions
    """
    
    # print("sim_word_ids_dict:", sim_word_ids_dict)
    # print("sim_values:", sim_values)
    # print("sub_word_dict:", sub_word_dict)

    new_doc_token_id_list, score = word_re.recover_document_greedy_rank_pos(
        post_sep.tolist(),
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


def main(args=None, dataset=None):
    if not args:
        parser = HfArgumentParser(Args)
        args = parser.parse_args()


    if args.model_type == "cat":
        model = Cat.from_pretrained(args.model_name_or_path).to(args.device)
    elif args.model_type == "dot":
        model = Dot.from_pretrained(args.model_name_or_path).to(args.device)
    else:
        raise ValueError(f"Model type '{args.model_type}' not recognised")

    tokenizer = model.tokenizer
    embeddings = model.model.base_model.embeddings.word_embeddings
    semantic_helper = SemanticHelper(embeddings, tokenizer)

    if dataset is None:
        corpus = irds.load(args.dataset_id)
    else:
        corpus = dataset


    word_re = BERTWordRecover(
        tokenizer,
        max_query_length=args.max_query_length,
        max_doc_length=args.max_doc_length,
        corpus=corpus
    )

    ranked_pairs = {}

    if not args.toy:

        ranked_pairs, args.score_lookup = get_topk(args, corpus, model)
        dataloader = []

        for qid, group in ranked_pairs.groupby("qid"):
            if len(group) < 10:
                continue
            group = group.sort_values(by="score", ascending=False)

            doc_a = group.iloc[10:]
            doc_b = group.iloc[:10]

            docid_a = doc_a["doc_id"].tolist()
            docid_b = doc_b["doc_id"].tolist()

            sequences_a = tokenizer(
                [doc_a.iloc[0]["query"]] * len(docid_a),
                doc_a["text"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            sequences_b = tokenizer(
                [doc_b.iloc[0]["query"]] * len(docid_b),
                doc_b["text"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,  # Truncate if exceeding model max length
                max_length=512
            )

            # scores_a = [group.iloc[-1]["score"]]  # Score for docid_a
            # scores_b = group.iloc[:-1]["score"].tolist()  # Scores for docid_b
            scores_a = doc_a["score"].tolist()
            scores_b = doc_b["score"].tolist()

            dataloader.append({
                "qid": str(qid),
                "docid_a": docid_a,
                "docid_b": docid_b,
                "sequences_a": sequences_a,
                "sequences_b": sequences_b,
                "scores_a": scores_a,
                "scores_b": scores_b
            })
        print("Initialization complete. Now move onto attack")


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


    attack_pair_docs = {}   # {qid_docid: text}
    attack_pair_scores = {} # {qid_docid: score}

    for batch in dataloader:
        attack_text_dict = {}
        attack_score_dict = {}

        qid = batch["qid"]
        # Prepare positive and negative sequences
        all_a_docs = [row for row in batch["sequences_a"]["input_ids"]]
        all_b_docs = [row for row in batch["sequences_b"]["input_ids"]]
        all_docs = all_a_docs + all_b_docs

        combined_tokens = tokenizer(
            [tokenizer.decode(ids) for ids in all_docs],
            padding=True,
            truncation=True,
            max_length=args.max_doc_length,
            return_tensors="pt",
        )

        ranking = {
            "input_ids": combined_tokens["input_ids"],
            "token_type_ids": combined_tokens["token_type_ids"],
            "position_ids": torch.arange(combined_tokens["input_ids"].size(-1))
            .unsqueeze(0)
            .repeat(combined_tokens["input_ids"].size(0), 1),
        }

        # Process each doc A in the batch
        num_doc_a = len(batch["docid_a"])
        for i in range(num_doc_a):
            doc_id_a = batch["docid_a"][i]
            sequence_score = batch["scores_a"][i]

            # Create a ranking slice for this positive document.
            # Assumption: the first num_doc_a rows in ranking correspond to sequences_a.
            single_ranking = {
                "input_ids": torch.cat((ranking["input_ids"][i].unsqueeze(0), ranking["input_ids"][num_doc_a:]), dim=0),
                "token_type_ids": torch.cat((ranking["token_type_ids"][i].unsqueeze(0), ranking["token_type_ids"][num_doc_a:]), dim=0),
                "position_ids": torch.cat((ranking["position_ids"][i].unsqueeze(0), ranking["position_ids"][num_doc_a:]), dim=0)
            }

            new_tokens_ids, score = attack(
                single_ranking,
                sequence_score,
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

            attack_text = tokenizer.decode(new_tokens_ids, skip_special_tokens=True)
            new_key = f"{qid}_{doc_id_a}"
            attack_text_dict[new_key] = attack_text
            attack_score_dict[new_key] = score
            attack_pair_docs[new_key] = attack_text
            attack_pair_scores[new_key] = score
            
            """
            Uncomment the following to write the result of attacking a single dataset
            """
            
            # Recover the original document for comparison:
            # input_ids = batch["sequences_a"]["input_ids"][i]
            # sep_indices = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            # first_sep_index = sep_indices[0].item()
            # The document tokens are everything after the first [SEP] token
            # doc_token_ids = input_ids[first_sep_index + 1:]
            # doc_text_decoded = tokenizer.decode(doc_token_ids, skip_special_tokens=True)
            #
            # recovered_attack_doc = word_re.tokenizer.decode(new_tokens_ids, skip_special_tokens=True)
            # same = recovered_attack_doc == doc_text_decoded
            # to_write = (
            #     f"key: {new_key}\n"
            #     f"recovered_attack_doc: {word_re.tokenizer.decode(new_tokens_ids, skip_special_tokens=True)}\n"
            #     f"ori_doc: {doc_text_decoded}\n"
            #     f"attack_score: {attack_score_dict[new_key]}\n"
            #     f"ori_score: {sequence_score}\n"
            #     f"same: {same}\n\n"
            # )
            #
            # if args.toy:
            #     print(to_write)
            # else:
            #     with open(args.output_file, "a") as f:
            #         f.write(to_write)

    return attack_pair_docs, args, ranked_pairs[["qid", "doc_id", "rank", "score"]], attack_pair_scores, tokenizer

if __name__ == "__main__":
    main()
