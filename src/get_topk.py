from rankers import CatTransformer
from pyterrier_pisa import PisaIndex
import ir_datasets as irds
from rankers._util import save_json
import pandas as pd
from fire import Fire


def get_topk(
    out_dir: str,
    dataset_id: str,
    index_path: str,
    model_name_or_path: str,
    k=10,
    subsample: int = 100,
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

    if subsample:
        queries = queries.sample(subsample)

    rez = pipeline.transform(queries)

    data = rez.groupby("qid").doc_id.apply(list).to_dict()

    # turn from dict to list of records
    data = [{"qid": qid, "doc_id_b": doc_id} for qid, doc_id in data.items()]
    formatted_model = model_name_or_path.replace("/", "-")
    formatted_dataset = dataset_id.replace("/", "-")
    out_file = f"{out_dir}/{formatted_model}-{formatted_dataset}-{k}.jsonl"

    save_json(data, out_file)

    return f"Saved to {out_file}"


if __name__ == "__main__":
    Fire(get_topk)
