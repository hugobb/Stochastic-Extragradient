import unittest
import torch
from hamiltonian.games import QuadraticGame
from hamiltonian.algorithms import ConsensusOptimization


class Test(unittest.TestCase):
    def test_games_opt(self):
        torch.manual_seed(1234)
        game = QuadraticGame(10, 10, mu=1, L=1, mu_B=1, L_B=1, bias=True, normal=True)
        alg = ConsensusOptimization(game, lr=1e-2, lr_H=1e-2, full_batch=False)
        results = alg.run(10)
        print("Quadratic Game Test: SUCCESSFUL")


if __name__ == '__main__':
    unittest.main()