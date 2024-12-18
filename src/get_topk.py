from rankers import CatTransformer
from pyterrier_pisa import PisaIndex
import ir_datasets as irds
from rankers._util import save_json
import pandas as pd
from fire import Fire


def get_topk(
    out_dir: str,
    dataset_id: str,
    pairs_file : str,
    index_path: str,
    model_name_or_path: str,
    k=10,
    subsample: int = -1,
    batch_size: int = 128,
):
    dataset = irds.load(dataset_id)
    index = PisaIndex(index_path, threads=4)
    bm25 = index.bm25() % 100

    model = (
        CatTransformer.from_pretrained(model_name_or_path, batch_size=batch_size) % k
    )

    pipeline = bm25 >> model

    queries = pd.DataFrame(dataset.queries_iter()).rename(
        columns={"query_id": "qid", "query": "query"}
    )

    pairs = pd.read_csv(pairs_file, sep="\t")

    # filter queries to the qids found in pairs qid
    queries = queries[queries.qid.isin(pairs.qid)]

    rez = pipeline.transform(queries)

    data = rez.groupby("qid").doc_id.apply(list).to_dict()
    scores = (
        rez.groupby("qid")
        .apply(lambda x: dict(zip(x["docno"], x["score"])))
        .to_dict()
    )

    # sort by score
    data = {qid: sorted(doc_ids, key=lambda x: scores[qid][x], reverse=True) for qid, doc_ids in data.items()}
    pairs['doc_id_b'] = pairs['qid'].map(data)

    formatted_model = model_name_or_path.replace("/", "-")
    formatted_dataset = dataset_id.replace("/", "-")
    out_file = f"{out_dir}/{formatted_model}-{formatted_dataset}-{k}.tsv.gz"
    score_out_file = f"{out_dir}/{formatted_model}-{formatted_dataset}-{k}-scores.json.gz"

    pairs.to_csv(out_file, sep="\t", index=False)
    save_json(scores, score_out_file)

    return f"Saved to {out_file}"


if __name__ == "__main__":
    Fire(get_topk)
