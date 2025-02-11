from rankers import CatTransformer
from pyterrier_pisa import PisaIndex
import ir_datasets as irds
from rankers._util import save_json
import pandas as pd


def get_topk(
    out_dir: str,
    dataset_id: str,
    model_name_or_path: str,
    index_path: str = "./index",
    k=10,
    batch_size: int = 128,
):
    dataset = irds.load(dataset_id)

    index = PisaIndex(index_path, threads=4)

    # Uncomment the following to create indexes

    # index.index(
    #     (
    #         {"docno": doc.doc_id, "text": doc.text}  # Correct mapping for PISA
    #         for doc in dataset.docs_iter()
    #     )
    # )

    bm25 = index.bm25() % 100

    model = (
        CatTransformer.from_pretrained(model_name_or_path, batch_size=batch_size) % k
    )

    # pipeline = bm25 >> model

    queries = pd.DataFrame(dataset.queries_iter()).rename(
        columns={"query_id": "qid", "text": "query"}
    )
    retrieved_docs = bm25.transform(queries)
    docs = pd.DataFrame(dataset.docs_iter()).rename(columns={"text": "doc_text"})
    retrieved_docs = retrieved_docs.rename(columns={"docno": "doc_id"})

    retrieved_docs = retrieved_docs.merge(docs, on="doc_id", how="left")
    retrieved_docs = retrieved_docs.dropna()
    retrieved_docs = retrieved_docs.rename(columns={"doc_text": "text"})
    print(retrieved_docs.head())
    print(docs.head())
    ranked_docs = model.transform(retrieved_docs)

    print(ranked_docs.head())

    scores = (
        ranked_docs.groupby("qid")
        .apply(lambda x: dict(zip(x["doc_id"], x["score"])))
        .to_dict()
    )

    # sort by score
    data = {
        qid: [doc_id for doc_id, _ in sorted(doc_score.items(), key=lambda item: item[1], reverse=True)[:k]]
        for qid, doc_score in scores.items()
    }

    # Add ranked documents
    ranked_docs["doc_id_b"] = ranked_docs["qid"].map(data)

    formatted_model = model_name_or_path.replace("/", "-")
    formatted_dataset = dataset_id.replace("/", "-")
    out_file = f"{out_dir}/{formatted_model}-{formatted_dataset}-{k}.tsv.gz"
    score_out_file = f"{out_dir}/{formatted_model}-{formatted_dataset}-{k}-scores.json.gz"

    ranked_docs.to_csv(out_file, sep="\t", index=False)
    save_json(scores, score_out_file)

    return ranked_docs, scores

