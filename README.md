# [Stochastic Gradient Descent-Ascent and Consensus Optimization for Smooth Games: Convergence Analysis under Expected Co-coercivity](https://arxiv.org/abs/2107.00052)

This repository is the official implementation of [Stochastic Gradient Descent-Ascent and Consensus Optimization for Smooth Games: Convergence Analysis under Expected Co-coercivity.](https://arxiv.org/abs/2107.00052)

## Installation

```
git clone https://github.com/hugobb/StochasticGamesOpt.git
cd StochasticGamesOpt
pip install .
```



## Notebooks
To reproduce the results of the paper simply open the notebook `hamiltonian/experiments/experiments_manager.ipynb`.

## Structure of the code
There is two main class `Game` and `Algorithm`.
To instantiate a quadratic game and run consensus optimization on it call:
```python
game = QuadraticGame(dim, n_samples)
alg = ConsensusOptimization(game, lr=lr, lr_H=lr)
results = alg.run(n_iter)
```
This will return a `dict` with different metrics.

## Citation
```
@misc{loizou2021stochastic,
    title={Stochastic Gradient Descent-Ascent and Consensus Optimization for Smooth Games: Convergence Analysis under Expected Co-coercivity},
    author={Nicolas Loizou and Hugo Berard and Gauthier Gidel and Ioannis Mitliagkas and Simon Lacoste-Julien},
    year={2021}
}
```
