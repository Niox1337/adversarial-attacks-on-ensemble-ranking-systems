from src import new_attack
import pandas as pd
from pyterrier_pisa import PisaIndex
import ir_datasets as irds

ranked_pairs, attack_ids_docs, args = new_attack.main()


dataset = irds.load(args.dataset_id)
attack_data = {doc.doc_id: doc.text for doc in dataset.docs_iter()}

for ids, text in attack_ids_docs.items():
    doc_id = ids.split("_")[1]
    attack_data[doc_id] = text

index = PisaIndex("./index", threads=4)
index.index(attack_data)
bm25 = index.bm25(num_results=100)

queries = pd.DataFrame(dataset.queries_iter()).rename(
    columns={"query_id": "qid", "text": "query"}
)
retrieved_docs = bm25.transform(queries)
retrieved_docs = retrieved_docs.rename(columns={"docno": "doc_id"})

retrieved_attack_num = 0
for ids in attack_ids_docs.keys():
    qid, doc_id = ids.split("_")[0], ids.split("_")[1]
    successful_attack = retrieved_docs[(retrieved_docs['qid'] == qid) & (retrieved_docs['doc_id'] == doc_id)]
    if not successful_attack.empty:
        retrieved_attack_num += 1

print(retrieved_attack_num)