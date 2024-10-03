# [Intriguing properties of neural networks](https://arxiv.org/pdf/1312.6199)

__Comparison__
- Our project
	- Aims to analyze the impact of adversarial attacks in a specific area
- This paper
	- Is the first discovery of adversarial attacks
	- Has no enough depth on adversarial attacks because it is a "by-finding" of another topic
##### Adversarial examples tend to be **robust** and **transferable**
- adversarial examples generated for one neural network are often still effective at fooling other neural networks, even when those networks:
	- Have different architectures
	- Were trained on different subsets of the data
	- Were trained with different hyperparameters
- This robustness of adversarial examples manifest some common fundamental weaknesses of different neural networks
##### Adversarial Negatives _could be_ minor cases
- This paper conjectured that adversarial negatives are extremely uncommon because neural networks have enough generalisation performances.
- This statement is just a guess because the researchers did not carry out the study on the cause of adversarial negative.

# [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572)

__Comparison__
- Our project
	- Is related to issues in practice
- This paper
	- Has the main topic of exploiting the cause of adversarial examples
##### **Linear behavior in high-dimensional spaces** is enough to reproduce adversarial examples.
- Adversarial training was developed to reduce the vulnerability of models to adversarial examples by providing more advanced regularisation method

##### Tension Between Linear and Nonlinear Models
- There's a trade-off:
	- **linear models** are easier to train and more efficient, but they are more vulnerable to adversarial examples
	- **nonlinear models** can be more resistant to attacks but are harder to train and optimise.

##### The root cause of Adversarial Examples
- The adversarial example is the side-effect of models being way too linear
- The perturbations are very similar to the weight vectors of the model.
- Different models use similar functions to train while the tasks are identical
- The direction of perturbation is the main impact to form a adversarial example
##### __Ensembles cannot defend adversarial examples__

# [Deep Text Classification Can be Fooled](https://arxiv.org/pdf/1704.08006)

**Comparison**
- Our Project:
	- Has the goal of evaluating the impact of adversarial attacks on search engine
- This paper:
	- Evaluates the tolerance of different kinds of data
	- Analyse how much will perturbed discrete data affects the performance of deep neural network
##### Tolerance of Perturbations
- Image and audio are both continuous data
- Text is discrete
- Continuous data has more tolerance to perturbed input than discrete one

# [Passage Re-Ranking with BERT](https://arxiv.org/pdf/1901.04085)
##### Process of using BERT to rank passages
1. Use a standard retrieval method (e.g. BM25) to fetch a large number of (potentially) relevant documents from a corpus
2. Re-Ranking with BERT: For each passage-document pair (query + passage), estimate how relevant the passage is to the query. This relevance is represented by a score.
	- Specifically, the model extracts a special **\[CLS] token** representation from BERT’s output, which will be the input to a single-layer neural network to estimate the probability of relevance.

# [PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models](https://arxiv.org/pdf/2204.01321)

##### WSRA
- Stands for **Word Substitution Ranking Attack**
- Aims to manipulate the rankings of search results by introducing **adversarial word substitutions**
- The objective is to promote documents' rankings by fooling **NRMs (Neural Ranking Models)** through adversarial attacks

##### PRADA
- Stands for **Pseudo-Relevance based ADversarial ranking Attack**
- Is a way of generating adversarial ranking attack


**WSRA focuses on manipulating search rankings using word substitutions, and PRADA offers a practical and effective method to carry out these attacks**

# [Topic-oriented Adversarial Attacks against Black-box Neural Ranking Models](https://dl.acm.org/doi/pdf/10.1145/3539618.3591777)

##### TARA
- Stands for **topic-oriented adversarial ranking attack**
- Fooling the neural network by rephrasing sentences/words in the document

##### RELEVANT
- Stands for reinforcement learning-based adversarial ranking attack
- Uses Markov decision process to update the attack strategy from previous findings

**Black-box ranking systems still has their vulnerabilities**