from torch.cpu import is_available

from src import new_attack
import pandas as pd
from pyterrier_pisa import PisaIndex
import ir_datasets as irds
from  pyterrier import rewrite
import argparse
from torch.cuda import is_available

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for attack evaluation.")

    parser.add_argument("--model_name_or_path", type=str, default="crystina-z/monoELECTRA_LCE_nneg31",
                        help="Path to the model or model identifier from HuggingFace.")
    parser.add_argument("--output_file", type=str, default="./output/attacks.txt",
                        help="Path to the output file where attack results will be stored.")
    parser.add_argument("--dataset", type=str, default="./dataset.tsv.gz",
                        help="Path to the dataset file.")
    parser.add_argument("--scores", type=str, default="./scores.json.gz",
                        help="Path to the scores file.")
    parser.add_argument("--model_type", type=str, default="cat",
                        help="Type of model to use.")
    parser.add_argument("--subset", type=int, default=1,
                        help="Subset size for evaluation.")
    parser.add_argument("--max_attack_word_number", type=int, default=10,
                        help="Maximum number of words to attack.")
    parser.add_argument("--toy", action="store_true",
                        help="Use toy mode (default: False).")
    parser.add_argument("--eps", type=float, default=0.1,
                        help="Epsilon value for attack algorithm.")
    parser.add_argument("--max_iter", type=int, default=3,
                        help="Maximum number of iterations for attack.")
    parser.add_argument("--simi_candi_topk", type=int, default=5,
                        help="Top-K similar words to consider.")
    parser.add_argument("--simi_threshod", type=float, default=0.4,
                        help="Similarity threshold for word substitution.")
    parser.add_argument("--max_doc_length", type=int, default=300,
                        help="Maximum document length.")
    parser.add_argument("--max_query_length", type=int, default=30,
                        help="Maximum query length.")
    parser.add_argument("--device", type=str, default="cuda" if is_available() else "cpu",
                        help="Device to run the model on (default: cuda if available, else cpu).")
    parser.add_argument("--find_word_search", action="store_true",
                        help="Enable word search for finding attacks (default: False).")
    parser.add_argument("--doc_list_length", type=int, default=100,
                        help="Number of documents in the list.")
    parser.add_argument("--dataset_id", type=str, default="msmarco-passage/trec-dl-2019/judged",
                        help="Dataset identifier.")
    parser.add_argument("--group_size", type=int, default=20,
                        help="Size of each evaluation group.")
    parser.add_argument("--rewrite_query", type=bool, default=True,
                        help="Enable rewriting of queries.")
    parser.add_argument("--multiple_retrievers", type=bool, default=False,
                        help="Enable multi-retrievers evaluation.")

    return parser.parse_args()


def evaluate_pipeline(pipeline, queries, attack_ids_docs, dataset_id, retriever, rewriter=None):

    retrieved_docs = pipeline.transform(queries).rename(columns={"docno": "doc_id"})

    # Evaluate attack success
    retrieved_attack_num = sum(
        (retrieved_docs["qid"] == ids.split("_")[0]) &
        (retrieved_docs["doc_id"] == ids.split("_")[1])
        for ids in attack_ids_docs.keys()
    )

    total_attacks = len(attack_ids_docs)
    attack_retrieval_ratio = retrieved_attack_num / total_attacks if total_attacks > 0 else 0

    evaluation_results.append({
        "dataset": dataset_id,
        "retriever": retriever,
        "rewriter": rewriter,
        "successful_attacks": retrieved_attack_num,
        "total_attacks": total_attacks,
        "attack_retrieval_ratio": attack_retrieval_ratio
    })

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
evaluation_results = []
args = parse_args()

for parent, children in datasets.items():
    args.first_child = True
    for child in children:
        args.dataset_id = f"{parent}/{child}"
        print(f"\nEvaluating dataset: {args.dataset_id}")

        # Load dataset
        dataset = irds.load(args.dataset_id)
        queries = pd.DataFrame(dataset.queries_iter()).rename(
            columns={"query_id": "qid", "text": "query"}
        )

        # Iterate over retrievers
        for retriever in retrievers:

            # Run attacks
            ranked_pairs, attack_ids_docs, args = new_attack.main(args, dataset)
            attack_data = {doc.doc_id: doc.text for doc in dataset.docs_iter()}

            # Inject attack documents into dataset
            for ids, text in attack_ids_docs.items():
                doc_id = ids.split("_")[1]
                attack_data[doc_id] = text

            # Index attack-modified dataset
            index = PisaIndex("./attack_index", threads=4)
            index.index(
                ({"docno": doc_id, "text": text} for doc_id, text in attack_data.items())
            )

            print(f"  - Evaluating retriever: {retriever}")

            # Retrieve documents using the selected retriever
            retriever_model = getattr(index, retriever)(num_results=100)
            args.pipeline = retriever_model
            evaluate_pipeline(retriever_model, queries, attack_ids_docs, args.dataset_id, retriever)

            #Retrieval with Rewriting
            if args.rewrite_query:
                for rewriter in rewriters:
                    print(f"  - Evaluating retriever with rewriter: {rewriter}")

                    rewriter_model = getattr(rewrite, rewriter)(index)
                    args.pipeline = retriever_model >> rewriter_model >> retriever_model

                    evaluate_pipeline(args.pipeline, queries, attack_ids_docs, args.dataset_id, retriever, rewriter)

            if args.multiple_retrievers:
                remaining_retrievers = retrievers.copy()
                remaining_retrievers.remove(retriever)
                for second_retriever in remaining_retrievers:
                    print(f"  - Evaluating pipeline with second retriever: {second_retriever}")

                    args.pipeline = retriever_model >> getattr(index, second_retriever)(num_results=100)
                    evaluate_pipeline(args.pipeline, queries, attack_ids_docs, args.dataset_id, second_retriever)


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