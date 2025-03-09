from rankers import CatTransformer
from pyterrier_pisa import PisaIndex
import ir_datasets as irds
from rankers._util import save_json
import pandas as pd


def get_topk(
    args,
    dataset,
    model,
    index_path: str = "./index",
    k=100,
    batch_size: int = 128,
):

    index = PisaIndex(index_path, threads=4)

    if args.first_child:
        index.index(
            (
                {"docno": doc.doc_id, "text": doc.text}  # Correct mapping for PISA
                for doc in dataset.docs_iter()
            )
        )

    bm25 = index.bm25(num_results=k)
    # pipeline = bm25 >> model
    model = model

    queries = pd.DataFrame(dataset.queries_iter()).rename(
        columns={"query_id": "qid", "text": "query"}
    )
    retrieved_docs = bm25.transform(queries)
    docs = pd.DataFrame(dataset.docs_iter()).rename(columns={"text": "doc_text"})
    retrieved_docs = retrieved_docs.rename(columns={"docno": "doc_id"})

    retrieved_docs = retrieved_docs.merge(docs, on="doc_id", how="left")
    retrieved_docs = retrieved_docs.dropna()
    retrieved_docs = retrieved_docs.rename(columns={"doc_text": "text"})
    ranked_docs = model.transform(retrieved_docs)


    scores = (
        ranked_docs.groupby("qid")
        .apply(lambda x: dict(zip(x["doc_id"], x["score"])))
        .to_dict()
    )

    # # sort by score
    # data = {
    #     qid: [doc_id for doc_id, _ in sorted(doc_score.items(), key=lambda item: item[1], reverse=True)[:k]]
    #     for qid, doc_score in scores.items()
    # }

    # Add ranked documents
    # ranked_docs["doc_id_b"] = ranked_docs["qid"].map(data)
    #
    # formatted_model = model_name_or_path.replace("/", "-")
    # formatted_dataset = dataset_id.replace("/", "-")
    # out_file = f"{out_dir}/{formatted_model}-{formatted_dataset}-{k}.tsv.gz"
    # score_out_file = f"{out_dir}/{formatted_model}-{formatted_dataset}-{k}-scores.json.gz"
    #
    # ranked_docs.to_csv(out_file, sep="\t", index=False)
    # save_json(scores, score_out_file)

    return ranked_docs, scores

