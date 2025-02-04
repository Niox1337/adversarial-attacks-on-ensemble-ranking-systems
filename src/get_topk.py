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

    pipeline = bm25 >> model

    queries = pd.DataFrame(dataset.queries_iter()).rename(
        columns={"query_id": "qid", "text": "query"}
    )

    docs = pd.DataFrame(dataset.docs_iter()).rename(columns={"doc_id": "docno"})

    pairs = pd.DataFrame([{"qid": qrel.query_id, "doc_id": qrel.doc_id} for qrel in dataset.qrels_iter() if qrel.relevance > 2])

    pairs = pairs.merge(docs, left_on="doc_id", right_on="docno", how="left")
    pairs = pairs.merge(queries, on="qid", how="left")
    pairs = pairs.dropna()
    # filter queries to the qids found in pairs qid


    rez = pipeline.transform(pairs)

    data = rez.groupby("qid").doc_id.apply(list).to_dict()
    scores = (
        rez.groupby("qid")
        .apply(lambda x: dict(zip(x["doc_id"], x["score"])))
        .to_dict()
    )

    # sort by score
    data = {qid: sorted(doc_ids, key=lambda x: scores[qid][x], reverse=True) for qid, doc_ids in data.items()}
    ranked_data = pd.DataFrame([
        {"qid": qid, "doc_id_b": doc_id, "rank": rank + 1}
        for qid, doc_ids in data.items()
        for rank, doc_id in enumerate(doc_ids)
    ])
    pairs = pairs.merge(ranked_data, on=["qid", "doc_id"], how="left")

    formatted_model = model_name_or_path.replace("/", "-")
    formatted_dataset = dataset_id.replace("/", "-")
    out_file = f"{out_dir}/{formatted_model}-{formatted_dataset}-{k}.tsv.gz"
    score_out_file = f"{out_dir}/{formatted_model}-{formatted_dataset}-{k}-scores.json.gz"

    pairs.to_csv(out_file, sep="\t", index=False)
    save_json(scores, score_out_file)

    return f"Saved to {out_file}"

