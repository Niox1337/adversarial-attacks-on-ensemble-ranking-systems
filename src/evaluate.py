from torch.cpu import is_available

from src import new_attack
import pandas as pd
from pyterrier_pisa import PisaIndex
import ir_datasets as irds
from  pyterrier import rewrite
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from torch.cuda import is_available

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
        default=True,
        metadata={"help": "Enable rewriting of queries."}
    )
    multiple_retrievers: bool = field(
        default=False,
        metadata={"help": "Enable multi-retrievers evaluation."}
    )


def evaluate_pipeline(pipeline, queries, attack_ids_docs, dataset_id, pipeline_name, ori_docs):

    retrieved_docs = pipeline.transform(queries).rename(columns={"docno": "doc_id"})

    retrieved_attack_num = 0
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

        evaluation_data["qid"].append(qid_part)
        evaluation_data["query"].append(query_text)
        evaluation_data["doc_id"].append(doc_id_part)
        evaluation_data["original_doc"].append(orig_text)
        evaluation_data["attack_doc"].append(attack_text)
        evaluation_data["pipeline"].append(pipeline_name)
        evaluation_data["retrieved"].append(retrieved)

    total_attacks = len(attack_ids_docs)
    attack_retrieval_ratio = retrieved_attack_num / total_attacks if total_attacks > 0 else 0

    evaluation_results["dataset"].append(dataset_id)
    evaluation_results["pipeline"].append(pipeline_name)
    evaluation_results["successful_attacks"].append(retrieved_attack_num)
    evaluation_results["total_attacks"].append(total_attacks)
    evaluation_results["attack_retrieval_ratio"].append(attack_retrieval_ratio)

retrievers = ["bm25", "dph", "pl2", "qld", "quantized"]
datasets = {
    "msmarco-document-v2": [
        "trec-dl-2019/judged",
        "trec-dl-2020/judged",
        "trec-dl-2021/judged",
        "trec-dl-2022/judged",
    ]
}
rewriters = ["bo1", "rm3"]
evaluation_data = {
    "qid": [],
    "query": [],
    "doc_id": [],
    "original_doc": [],
    "attack_doc": [],
    "pipeline": [],
    "retrieved": [],
}
evaluation_results = {
    "dataset": [],
    "pipeline": [],
    "successful_attacks": [],
    "total_attacks": [],
    "attack_retrieval_ratio": [],
}


def main():
    parser = HfArgumentParser(AttackEvalArguments)
    args = parser.parse_args()

    for parent, children in datasets.items():
        args.first_child = True
        for child in children:
            args.dataset_id = f"{parent}/{child}"
            print(f"\nEvaluating dataset: {args.dataset_id}")

            dataset = irds.load(args.dataset_id)
            print("Constructing original_docs")

            queries = pd.DataFrame(dataset.queries_iter()).rename(
                columns={"query_id": "qid", "text": "query"}
            )
            print("Constructing original_docs")
            original_docs = {doc.doc_id: doc.body for doc in dataset.docs_iter()}

            for retriever in retrievers:
                print("Start Attacking")
                attack_ids_docs, args = new_attack.main(args)
                print(f"  - Evaluating retriever: {retriever}")
                attack_data = original_docs.copy()

                for ids, text in attack_ids_docs.items():
                    doc_id = ids.split("_")[1]
                    attack_data[doc_id] = text

                index = PisaIndex("./attack_index", threads=4)
                index.index(
                    ({"docno": doc_id, "text": text} for doc_id, text in attack_data.items())
                )


                retriever_model = getattr(index, retriever)(num_results=100)
                args.pipeline = retriever_model
                evaluate_pipeline(retriever_model, queries, attack_ids_docs, args.dataset_id, retriever, original_docs)
                args.first_child = False
                if args.rewrite_query:
                    for rewriter in rewriters:
                        print(f"  - Evaluating retriever with rewriter: {rewriter}")

                        rewriter_model = getattr(rewrite, rewriter)(index)
                        args.pipeline = retriever_model >> rewriter_model >> retriever_model

                        evaluate_pipeline(args.pipeline, queries, attack_ids_docs, args.dataset_id, retriever+" -> "+rewriter+" -> "+retriever, original_docs)

                if args.multiple_retrievers:
                    remaining_retrievers = retrievers.copy()
                    remaining_retrievers.remove(retriever)
                    for second_retriever in remaining_retrievers:
                        print(f"  - Evaluating pipeline: {retriever} -> {second_retriever}")

                        args.pipeline = retriever_model >> getattr(index, second_retriever)(num_results=100)
                        evaluate_pipeline(args.pipeline, queries, attack_ids_docs, args.dataset_id, retriever + " -> " + second_retriever, original_docs)

    evaluation_data_df = pd.DataFrame(evaluation_data)
    evaluation_results_df = pd.DataFrame(evaluation_results)

    evaluation_data_df.to_csv("./output/evaluation_data.csv", index=False)
    evaluation_results_df.to_csv("./output/evaluation_results.csv", index=False)

if __name__ == "__main__":
    main()

# ranked_pairs, attack_ids_docs, args = new_attack.main()
#
# dataset = irds.load(args.dataset_id)
# attack_data = {doc.doc_id: doc.text for doc in dataset.docs_iter()}
#
# for ids, text in attack_ids_docs.items():
#     doc_id = ids.split("_")[1]
#     attack_data[doc_id] = text
#
# index = PisaIndex("./attack_index", threads=4)
#
# index.index(
#     (
#         {"docno": doc_id, "text": text}
#         for doc_id, text in attack_data.items()
#     )
# )
#
# bm25 = index.bm25(num_results=100)
#
# queries = pd.DataFrame(dataset.queries_iter()).rename(
#     columns={"query_id": "qid", "text": "query"}
# )
# retrieved_docs = bm25.transform(queries)
# retrieved_docs = retrieved_docs.rename(columns={"docno": "doc_id"})
#
# retrieved_attack_num = 0
# for ids in attack_ids_docs.keys():
#     qid, doc_id = ids.split("_")[0], ids.split("_")[1]
#     successful_attack = retrieved_docs[(retrieved_docs['qid'] == qid) & (retrieved_docs['doc_id'] == doc_id)]
#     if not successful_attack.empty:
#         retrieved_attack_num += 1
#
# print("Number of successful attacks: ", len(attack_ids_docs))
# print("Retrieved attacks: ", retrieved_attack_num)
# print("Ratio: ", retrieved_attack_num/len(attack_ids_docs))