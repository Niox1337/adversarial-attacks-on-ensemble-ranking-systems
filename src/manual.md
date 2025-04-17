# ⚙Configuration Manual

This section documents all configurable parameters used in the adversarial attack pipeline. These parameters can be passed via command line or configuration files.

---

### File and Model Paths

- **`model_name_or_path`** (str)  
  *Default*: `"crystina-z/monoELECTRA_LCE_nneg31"`  
  Path to the pretrained model or HuggingFace model identifier.

- **`index_path`** (str)  
  *Default*: `"./index/"`  
  Path to the index files (for retrieval).

---

### Model Settings

- **`model_type`** (str)  
  *Default*: `"cat"`  
  Type of model to use. Can be task-specific or architecture-specific.

- **`device`** (str)  
  *Default*: `"cuda"` if available, otherwise `"cpu"`  
  Device to run the model on.

---

### Dataset and Evaluation Control

- **`dataset_id`** (str)  
  *Default*: `"msmarco-passage/trec-dl-2019/judged"`  
  Identifier for the dataset used.

- **`doc_list_length`** (int)  
  *Default*: `100`  
  Total number of documents per query used for ranking.

- **`toy`** (bool)  
  *Default*: `False`  
  Enable toy mode for quick debugging with a small subset.

---

### Adversarial Attack Parameters

- **`max_attack_word_number`** (int)  
  *Default*: `10`  
  Maximum number of words to perturb in a document/query.

- **`eps`** (float)  
  *Default*: `0.1`  
  Controls the step size of the document recovery algorithm.

- **`max_iter`** (int)  
  *Default*: `3`  
  Maximum number of iterations for the attack algorithm.

- **`find_word_search`** (bool)  
  *Default*: `False`  
  Whether to use an exhaustive word search to find optimal attacks.

---

### Semantic Constraints

- **`simi_candi_topk`** (int)  
  *Default*: `5`  
  Top-K most similar words considered for substitution.

- **`simi_threshod`** (float)  
  *Default*: `0.4`  
  Minimum semantic similarity threshold required for word substitution.

---

### Text Length Constraints

- **`max_doc_length`** (int)  
  *Default*: `300`  
  Maximum length of the document (in tokens or words).

- **`max_query_length`** (int)  
  *Default*: `30`  
  Maximum length of the query.

---

### Under Development

- **`rewrite_query`** (bool)  
  *Default*: `False`  
  Whether to apply query rewriting during attack.

- **`multiple_retrievers`** (bool)  
  *Default*: `False`  
  Whether to evaluate with multiple retrievers.

---
