from fire import Fire 
import ir_dataset as irds 
import pandas as pd

def get_subsample(dataset_id: str, output_directory : str, document_subsample: int = 100, query_subsample: int = 10):
    dataset = irds.load(dataset_id)
    qrels = pd.DataFrame(dataset.qrels_iter())
    assert len(qrels.query_id.unique()) <= query_subsample, "Query subsample is smaller than the number of unique queries in the dataset"

    qrels = qrels.groupby("relevance").apply(lambda x: x.sample(document_subsample)).reset_index(drop=True)

    qrels = qrels.groupby("query_id").doc_id.sample(1).reset_index()
    qrels.columns = ["qid", "doc_id_a"]

    qrels.to_csv(output_directory, sep="\t", index=False)

    return f"Saved to {output_directory}"

if __name__ == "__main__":
    Fire(get_subsample)
