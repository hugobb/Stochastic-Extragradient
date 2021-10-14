from .game import Game
import torch
from torch import nn
import math
from .sampler import RandomSampler
from numpy import linalg


class StochasticBilinearGame(Game):
    def __init__(self, dim, matrix=None, bias=True):
        super(StochasticBilinearGame, self).__init__()
        self.dim = dim

        self.matrix = matrix
        if self.matrix is None:
            self.matrix = torch.zeros(self.dim, self.dim, self.dim)
            for i in range(self.dim):
                self.matrix[i, i, i] = 1

        n_samples = len(self.matrix)
        self.sampler = RandomSampler(n_samples)

        self.a = 1.0 / math.sqrt(self.dim) * torch.zeros(n_samples, self.dim).normal_()
        self.b = 1.0 / math.sqrt(self.dim) * torch.zeros(n_samples, self.dim).normal_()
        if not bias:
            self.a = self.a.zero_()
            self.b = self.b.zero_()

        self.x_star, self.y_star = self.optimum()

        self.players = nn.ModuleList(
            [
                nn.ParameterList([nn.Parameter(torch.zeros(self.dim))]),
                nn.ParameterList([nn.Parameter(torch.zeros(self.dim))]),
            ]
        )
        self.init()

    def init(self):
        self.players[0][0].data = (
            1.0 / math.sqrt(self.dim) * self.players[0][0].data.normal_() * 10
        )
        self.players[1][0].data = (
            1.0 / math.sqrt(self.dim) * self.players[1][0].data.normal_() * 10
        )

    def loss(self, x=None):
        if x is None:
            x = self.sampler.sample_batch()

        loss = (
            (
                self.players[0][0].view(1, -1)
                * (self.matrix[x] * self.players[1][0].view(1, 1, -1)).sum(-1)
                + self.a[x] * self.players[0][0].view(1, -1)
                + self.b[x] * self.players[1][0].view(1, -1)
            )
            .sum(-1)
            .mean()
        )

        return [loss, -loss]

    def dist2opt(self):
        d = ((self.players[0][0].data.cpu() - self.x_star) ** 2).sum() + (
            (self.players[1][0].data.cpu() - self.y_star) ** 2
        ).sum()
        return d

    def optimum(self):
        matrix = self.matrix.mean(0)
        x = linalg.solve(matrix.T, -self.b.mean(0))
        y = linalg.solve(matrix, -self.a.mean(0))

        import numpy as np

        matrix = matrix.numpy()
        matrix = np.block(
            [[np.zeros_like(matrix), matrix], [-matrix.T, np.zeros_like(matrix)]]
        )
        b = np.concatenate([self.a, -self.b], axis=-1).mean(0)
        sol = linalg.solve(matrix, -b)
        x_star, y_star = np.split(sol, 2)
        print(x, y)
        print(x_star, y_star)

        return torch.tensor(x), torch.tensor(y)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.matrix = self.matrix.to(*args, **kwargs)
        self.a = self.a.to(*args, **kwargs)
        self.b = self.b.to(*args, **kwargs)
        return self


class GaussianBilinearGame(StochasticBilinearGame):
    def __init__(self, n_samples, dim, bias=True):
        matrix = torch.randn(n_samples, dim, dim)
        super().__init__(dim, matrix, bias)
