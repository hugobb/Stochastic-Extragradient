# Stochastic Extragradient:  General Analysis and Improved Rates

This repository is the official implementation of the paper "Stochastic Extragradient:  General Analysis and Improved Rates".

## Installation

```
git clone https://github.com/hugobb/Stochastic-Extragradient.git
cd Stochastic-Extragradient
pip install .
```



## Notebooks
To reproduce the results of the paper simply open the notebook `gamesopt/experiments/Exp1 AISTATS 2022.ipynb`.

## Structure of the code
There is two main class `Game` and `Algorithm`.
To instantiate a quadratic game and run stochastic extragradient (SEG) with same sample:
```python
game = QuadraticGame(dim, n_samples)
alg = SEG(game, lr=lr, lr_e=lr_e, same_sample=True)
results = alg.run(n_iter)
```
This will return a `dict` with different metrics.
