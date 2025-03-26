import os
from dataclasses import dataclass, field

import ir_datasets as irds
import pandas as pd
from pyterrier import rewrite
from pyterrier_pisa import PisaIndex
from torch.cuda import is_available
from transformers import HfArgumentParser

from src import new_attack


@dataclass
class AttackEvalArguments:
    model_name_or_path: str = field(
        default="crystina-z/monoELECTRA_LCE_nneg31",
        metadata={"help": "Path to the model or model identifier from HuggingFace."}
    )
    output_file: str = field(
        default="./output/attacks.txt",
        metadata={"help": "Path to the output file where attack results will be stored."}
    )
    dataset: str = field(
        default="./dataset.tsv.gz",
        metadata={"help": "Path to the dataset file."}
    )
    scores: str = field(
        default="./scores.json.gz",
        metadata={"help": "Path to the scores file."}
    )
    model_type: str = field(
        default="cat",
        metadata={"help": "Type of model to use."}
    )
    subset: int = field(
        default=1,
        metadata={"help": "Subset size for evaluation."}
    )
    max_attack_word_number: int = field(
        default=10,
        metadata={"help": "Maximum number of words to attack."}
    )
    toy: bool = field(
        default=False,
        metadata={"help": "Use toy mode."}
    )
    eps: float = field(
        default=0.1,
        metadata={"help": "Epsilon value for attack algorithm."}
    )
    max_iter: int = field(
        default=3,
        metadata={"help": "Maximum number of iterations for attack."}
    )
    simi_candi_topk: int = field(
        default=5,
        metadata={"help": "Top-K similar words to consider."}
    )
    simi_threshod: float = field(
        default=0.4,
        metadata={"help": "Similarity threshold for word substitution."}
    )
    max_doc_length: int = field(
        default=300,
        metadata={"help": "Maximum document length."}
    )
    max_query_length: int = field(
        default=30,
        metadata={"help": "Maximum query length."}
    )
    device: str = field(
        default="cuda" if is_available() else "cpu",
        metadata={"help": "Device to run the model on (default: cuda if available, else cpu)."}
    )
    find_word_search: bool = field(
        default=False,
        metadata={"help": "Enable word search for finding attacks."}
    )
    doc_list_length: int = field(
        default=100,
        metadata={"help": "Number of documents in the list."}
    )
    dataset_id: str = field(
        default="msmarco-passage/trec-dl-2019/judged",
        metadata={"help": "Dataset identifier."}
    )
    group_size: int = field(
        default=20,
        metadata={"help": "Size of each evaluation group."}
    )
    rewrite_query: bool = field(
        default=False,
        metadata={"help": "Enable rewriting of queries."}
    )
    multiple_retrievers: bool = field(
        default=False,
        metadata={"help": "Enable multi-retrievers evaluation."}
    )
    index_path: str = field(
        default="./index/",
        metadata={"help": "Path to the index files."}
    )


def evaluate_pipeline(pipeline, queries, attack_ids_docs, dataset_id, pipeline_name, ori_docs, ori_pairs,
                      attack_text_dict, attack_pair_scores):
    retrieved_docs = pipeline.transform(queries).rename(columns={"docno": "doc_id"})

    retrieved_docs["text"] = retrieved_docs["doc_id"].map(ori_docs)
    retrieved_docs["text"] = retrieved_docs["doc_id"].map(attack_text_dict)

    retrieved_attack_num = 0
    top10_num = 0
    for attack_key, attack_text in attack_ids_docs.items():
        qid_part, doc_id_part = attack_key.split("_")
        query_text = queries.loc[queries["qid"] == qid_part, "query"]
        query_text = query_text.iloc[0] if not query_text.empty else None
        orig_text = ori_docs.get(doc_id_part, None)
        # Check if the attack document was retrieved: i.e. whether there exists a row in retrieved_docs
        # with matching qid and doc_id.
        retrieved = ((retrieved_docs["qid"] == qid_part) & (retrieved_docs["doc_id"] == doc_id_part)).any()
        if retrieved:
            retrieved_attack_num += 1

        ori_pair = ori_pairs.loc[(ori_pairs["qid"] == qid_part) & (ori_pairs["doc_id"] == doc_id_part)]
        ori_rank = ori_pair["rank"].values[0]
        # ori_score = ori_pair["score"].values[0]
        val = attack_pair_scores[attack_key]
        new_score = val if isinstance(val, float) else val[0]
        if retrieved:
            original_subset = ori_pairs.loc[ori_pairs["qid"] == qid_part]
            # rank starts at 0
            new_rank = (original_subset["score"] > new_score).sum()
        else:
            new_rank = -1

        if new_rank != -1 and new_rank < 10:
            top10_num += 1
            evaluation_data["top10"].append(True)
        else:
            evaluation_data["top10"].append(False)
        improvement = (ori_rank - new_rank) if new_rank != -1 else None

        evaluation_data["original_rank"].append(ori_rank)
        evaluation_data["attacked_rank"].append(new_rank)
        evaluation_data["qid"].append(qid_part)
        evaluation_data["query"].append(query_text)
        evaluation_data["doc_id"].append(doc_id_part)
        evaluation_data["original_doc"].append(orig_text)
        evaluation_data["attack_doc"].append(attack_text)
        evaluation_data["pipeline"].append(pipeline_name)
        evaluation_data["retrieved"].append(retrieved)
        evaluation_data["rank_improvement"].append(improvement)

    total_attacks = len(attack_ids_docs)
    attack_retrieval_ratio = retrieved_attack_num / total_attacks if total_attacks > 0 else 0

    evaluation_results["dataset"].append(dataset_id)
    evaluation_results["pipeline"].append(pipeline_name)
    evaluation_results["successful_attacks"].append(retrieved_attack_num)
    evaluation_results["total_attacks"].append(total_attacks)
    evaluation_results["attack_retrieval_ratio"].append(attack_retrieval_ratio)
    evaluation_results["top10_num"].append(top10_num)


def remove_files_in_directory(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print(f'All files in "{directory_path}" have been removed.')
    else:
        print(f'Directory "{directory_path}" does not exist.')


retrievers = ["bm25", "dph", "pl2", "qld"]
datasets = {
    "msmarco-passage": [
        "trec-dl-2019/judged",
        "trec-dl-2020/judged",
    ],
    "cord19": [
        "trec-covid"
    ],
    "cord19/fulltext": [
        "trec-covid",
    ]
}
# retrievers = ["bm25"]
# datasets = {
#     "msmarco-passage": [
#         "trec-dl-2019/judged",
#     ]
# }
# datasets = {
#     "cord19": [
#         "trec-covid"
#     ],
#     "cord19/fulltext": [
#         "trec-covid",
#     ]
# }
rewriters = ["Bo1QueryExpansion", "RM3"]
evaluation_data = {
    "qid": [],
    "query": [],
    "doc_id": [],
    "original_doc": [],
    "attack_doc": [],
    "pipeline": [],
    "retrieved": [],
    "original_rank": [],
    "attacked_rank": [],
    "rank_improvement": [],
    "top10": []
}
evaluation_results = {
    "dataset": [],
    "pipeline": [],
    "successful_attacks": [],
    "total_attacks": [],
    "attack_retrieval_ratio": [],
    "top10_num": []
}


def main():
    parser = HfArgumentParser(AttackEvalArguments)
    args = parser.parse_args()

    for parent, children in datasets.items():
        args.first_child = True
        docs_dataset = irds.load(parent)
        print("Constructing original_docs")
        if not "cord19" in parent:
            original_docs = {doc.doc_id: doc.text for doc in docs_dataset.docs_iter()}
        else:
            if "fulltext" in parent:
                original_docs = {doc.doc_id: doc.abstract + "\n" + doc.body[0].title for doc
                                 in docs_dataset.docs_iter() if doc.abstract}
            else:
                original_docs = {doc.doc_id: doc.abstract for doc in docs_dataset.docs_iter() if doc.abstract}
        for child in children:
            args.dataset_id = f"{parent}/{child}"
            print(f"\nEvaluating dataset: {args.dataset_id}")

            dataset = irds.load(args.dataset_id)

            print("Constructing queries")

            if not "cord19" in parent:
                queries = pd.DataFrame(dataset.queries_iter()).rename(
                    columns={"query_id": "qid", "text": "query"}
                )
            else:
                queries = pd.DataFrame(dataset.queries_iter()).rename(
                    columns={"query_id": "qid", "title": "query"}
                )

            for retriever in retrievers:
                print("Start Attacking")
                args.pipeline = [retriever]
                attack_ids_docs, args, ori_pairs, attack_pair_scores, tokenizer = new_attack.main(args)
                print(f"  - Evaluating retriever: {retriever}")
                attack_data = original_docs.copy()

                special_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens)
                for ids, text in attack_ids_docs.items():
                    query_id, doc_id = ids.split("_", 1)

                    query = queries.loc[queries["qid"] == query_id]["query"].values[0]
                    num_query_tokens = len(tokenizer.tokenize(query))

                    input_ids = tokenizer.encode(query, text, add_special_tokens=True)
                    num_specials = sum(1 for tok_id in input_ids if tok_id in special_token_ids)
                    doc_tok = tokenizer.tokenize(original_docs[doc_id])
                    doc_token_budget = args.max_doc_length-num_query_tokens-num_specials-1

                    if len(doc_tok)>doc_token_budget:
                        truncated = doc_tok[doc_token_budget:]
                        truncated = tokenizer.convert_tokens_to_string(truncated)
                        attacked_doc = text.strip() + " " + truncated.strip()
                        attack_data[doc_id] = attacked_doc
                        attack_ids_docs[ids] = attacked_doc
                    else:
                        attack_data[doc_id] = text

                index_path = args.index_path + "attack/" + args.dataset_id + "/" + retriever
                index = PisaIndex(index_path, threads=32)

                try:
                    remove_files_in_directory(index_path)
                    index.index(
                        ({"docno": doc_id, "text": text} for doc_id, text in attack_data.items())
                    )
                except RuntimeError:
                    print(f"{parent} attack document is already indexed, skipping...")

                retriever_model = getattr(index, retriever)(num_results=100, threads=32)
                evaluate_pipeline(retriever_model, queries, attack_ids_docs, args.dataset_id, retriever, original_docs,
                                  ori_pairs, attack_data, attack_pair_scores)
                args.first_child = False
                if args.rewrite_query:
                    for rewriter in rewriters:
                        print(f"  - Evaluating retriever with rewriter: {rewriter}")

                        rewriter_model = getattr(rewrite, rewriter)(index)
                        pipeline = retriever_model >> rewriter_model >> retriever_model
                        args.pipeline = [retriever, rewriter, retriever]

                        evaluate_pipeline(pipeline, queries, attack_ids_docs, args.dataset_id,
                                          retriever + " -> " + rewriter + " -> " + retriever, original_docs, ori_pairs,
                                          attack_data, attack_pair_scores)

                if args.multiple_retrievers:
                    remaining_retrievers = retrievers.copy()
                    remaining_retrievers.remove(retriever)
                    for second_retriever in remaining_retrievers:
                        print(f"  - Evaluating pipeline: {retriever} -> {second_retriever}")

                        pipeline = retriever_model >> getattr(index, second_retriever)(num_results=100)
                        args.pipeline = [retriever, second_retriever]
                        evaluate_pipeline(pipeline, queries, attack_ids_docs, args.dataset_id,
                                          retriever + " -> " + second_retriever, original_docs, ori_pairs, attack_data,
                                          attack_pair_scores)


    evaluation_data_df = pd.DataFrame(evaluation_data)
    evaluation_results_df = pd.DataFrame(evaluation_results)

    evaluation_results_df.to_csv("./output/evaluation_results.csv", index=False)

    with open("./output/evaluation_data.json", "w", encoding="utf-8") as file:
        file.write(evaluation_data_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
