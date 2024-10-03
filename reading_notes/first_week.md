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

# [Passage Re-Ranking with BERT](https://arxiv.org/abs/1901.04085)

# [PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models](https://arxiv.org/pdf/2204.01321)

# [Topic-oriented Adversarial Attacks against Black-box Neural Ranking Models](https://dl.acm.org/doi/pdf/10.1145/3539618.3591777)