from rankers import CatTransformer
from pyterrier_pisa import PisaIndex
import ir_datasets as irds
from rankers._util import save_json
import pandas as pd
from pyterrier import rewrite


def get_topk(
    args,
    dataset,
    model,
    index_path: str = "./index",
    k=100,
    batch_size: int = 128,
):
    print("Initializing attack dataset...")
    doc_name = args.dataset_id.split("/")[0]
    index_path = args.index_path + "original/" + doc_name
    index = PisaIndex(index_path, threads=32)

    if args.first_child:
        try:
            if "cord19" not in doc_name:
                index.index(
                    (
                        {"docno": doc.doc_id, "text": doc.body}
                        for doc in dataset.docs_iter()
                    )
                )
            else:
                if "fulltext" in doc_name:
                    index.index(
                        (
                            {"docno": doc.doc_id,
                             "text": doc.abstract + "\n" + doc.body[0].title + "\n" + doc.body[0].text}
                            for doc in dataset.docs_iter()
                        )
                    )
                else:
                    index.index(
                        (
                            {"docno": doc.doc_id, "text": doc.abstract}
                            for doc in dataset.docs_iter()
                        )
                    )
        except RuntimeError:
            print(f"{doc_name} original document is already indexed, skipping...")

    retrieval_pipeline = getattr(index, args.pipeline[0])(num_results=100, threads=32)
    if len(args.pipeline) > 1:
        for component in args.pipeline[1:]:
            if component in ["Bo1QueryExpansion", "RM3"]:
                retrieval_pipeline = retrieval_pipeline >> getattr(rewrite, component)(index)
            else:
                retrieval_pipeline = retrieval_pipeline >> getattr(index, component)()

    # pipeline = bm25 >> model
    model = (
            CatTransformer.from_pretrained(args.model_name_or_path, batch_size=batch_size)
    )

    if "cord19" not in doc_name:
        queries = pd.DataFrame(dataset.queries_iter()).rename(
            columns={"query_id": "qid", "text": "query"}
        )
    else:
        queries = pd.DataFrame(dataset.queries_iter()).rename(
            columns={"query_id": "qid", "title": "query"}
        )
    retrieved_docs = retrieval_pipeline.transform(queries)

    if "cord19" not in doc_name:
        docs = pd.DataFrame(dataset.docs_iter()).rename(columns={"body": "doc_text"})
    elif "fulltext" in doc_name:
        docs = pd.DataFrame({
            "doc_id": [doc.doc_id for doc in dataset.docs_iter()],
            "doc_text": [doc.abstract + "\n" + doc.body[0].title + "\n" + doc.body[0].text for doc in
                         dataset.docs_iter()]
        })
    else:
        docs = pd.DataFrame({
            "doc_id": [doc.doc_id for doc in dataset.docs_iter()],
            "doc_text": [doc.abstract for doc in dataset.docs_iter()]
        })

    retrieved_docs = retrieved_docs.rename(columns={"docno": "doc_id"})

    retrieved_docs = retrieved_docs.merge(docs, on="doc_id", how="left")
    retrieved_docs = retrieved_docs.dropna()
    retrieved_docs = retrieved_docs.rename(columns={"doc_text": "text"})
    retrieved_docs = retrieved_docs[retrieved_docs["text"].notnull() & (retrieved_docs["text"].str.strip() != "")]
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

