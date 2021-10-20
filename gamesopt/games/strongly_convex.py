from .game import Game
import torch
from torch import nn
from .sampler import RandomSampler
import math
import numpy.linalg as linalg
import numpy as np
from .utils import (
    make_commutative_matrix,
    make_random_matrix,
    make_sym_matrix,
)


class ScPLplusGame(Game):
    def __init__(self):
        super().__init__()

        self.dim = 1
        self.num_samples = 1
        self.sampler = RandomSampler(self.num_samples)

        self.players = nn.ModuleList(
            [nn.ParameterList([nn.Parameter(torch.zeros(self.dim))])]
        )

        self.init()

    def init(self):
        self.players[0][0].data = 10 * torch.ones(self.dim)

    def loss(self, x=None):
        if x is None:
            x = self.sampler.sample_batch()

        loss = 0.5 * (self.players[0][0]) ** 2 + abs(self.players[0][0])

        return [loss, -loss]

    def dist2opt(self):
        d = ((self.players[0][0]) ** 2).sum()
        return d


class QuadraticOptimization(Game):
    def __init__(self, dim, num_samples=1, mu=1.0, L=10.0):
        super(QuadraticOptimization, self).__init__()

        self.dim = dim
        self.num_samples = num_samples
        self.sampler = RandomSampler(num_samples)

        self.matrix, self.eigs = make_sym_matrix(num_samples, self.dim, mu, L)
        self.L = torch.symeig(self.matrix)[0].max()
        self.mu = torch.symeig(self.matrix.mean(0))[0].min()

        self.x_star = None

        self.players = nn.ModuleList(
            [nn.ParameterList([nn.Parameter(torch.zeros(self.dim))])]
        )

        self.init()

    def init(self):
        self.players[0][0].data = (
            1.0 / math.sqrt(self.dim) * torch.zeros(self.dim).normal_()
        )

    def loss(self, x=None):
        if x is None:
            x = self.sampler.sample_batch()

        loss = (
            (
                self.players[0][0].view(1, -1)
                * (self.matrix[x] * self.players[0][0].view(1, 1, -1)).sum(-1)
            )
            .sum(-1)
            .mean()
        )

        return [loss, -loss]

    def dist2opt(self):
        if self.x_star is None:
            self.x_star = self.optimum()
        d = ((self.players[0][0] - self.x_star) ** 2).sum()
        return d

    def optimum(self):
        x = torch.zeros_like(self.players[0][0])
        return x


class QuadraticGame(Game):
    def __init__(
        self,
        dim,
        num_samples=1,
        mu=0.0,
        L=1.0,
        mu_B=0.0,
        L_B=1.0,
        num_zeros=0,
        A=None,
        B=None,
        C=None,
        bias=False,
        normal=True,
        init_func=None,
    ):
        super().__init__()

        self.dim = dim
        self.num_samples = num_samples
        self.sampler = RandomSampler(num_samples)

        self.matrix = torch.zeros(3, num_samples, dim, dim)
        if A is None:
            self.matrix[0] = make_sym_matrix(num_samples, self.dim, mu, L, num_zeros)
        else:
            self.matrix[0] = A

        if C is None:
            self.matrix[1] = make_sym_matrix(num_samples, self.dim, mu, L, num_zeros)
        else:
            self.matrix[1] = C

        if B is None:
            self.matrix[2] = make_random_matrix(
                num_samples, self.dim, mu_B, L_B, normal=normal
            )
        else:
            self.matrix[2] = B

        self.bias = torch.zeros(3, num_samples, dim)
        if bias:
            self.bias = 1.0 / math.sqrt(self.dim) * self.bias.normal_() / 10

        self.x_star, self.y_star = self.optimum()

        self.players = nn.ModuleList(
            [
                nn.ParameterList([nn.Parameter(torch.zeros(self.dim))]),
                nn.ParameterList([nn.Parameter(torch.zeros(self.dim))]),
            ]
        )

        J = torch.zeros(num_samples, 2 * dim, 2 * dim)
        J[:, :dim, :dim] = self.matrix[0]
        J[:, :dim, dim:] = self.matrix[2]
        J[:, dim:, :dim] = -self.matrix[2].transpose(-2, -1)
        J[:, dim:, dim:] = self.matrix[1]

        eigvals = torch.linalg.eigvals(J.mean(0))

        self.L_i = torch.linalg.svdvals(J)[:, 0]
        self.L_max = self.L_i.max().item()
        self.L_mean = self.L_i.mean().item()
        self.L = torch.linalg.svdvals(J.mean(0))[0].item()
        self.mu_i = torch.linalg.eigvals(J).real.min(dim=-1)[0]
        self.mu_min = self.mu_i.min().item()
        self.mu = eigvals.real.min().item()
        self.imag_min = eigvals.imag.min().item()
        self.imag_max = eigvals.imag.max().item()

        mask = self.mu_i >= 0
        self.mu_mean = (((mask*self.mu_i).sum() + 4*((~mask)*self.mu_i).sum())/len(self.mu_i)).item()

        self.init_func = init_func

        self.init()

    def reset(self):
        if self.init_func is None:
            self.players[0][0].data = (
                1.0 / math.sqrt(self.dim) * self.players[0][0].data.normal_()
            )
            self.players[1][0].data = (
                1.0 / math.sqrt(self.dim) * self.players[1][0].data.normal_()
            )
        else:
            self.init_func(self.players)

    def loss(self, x=None):
        if x is None:
            x = self.sampler.sample_batch()

        loss = (
            (
                self.players[0][0].view(1, -1)
                * (self.matrix[0, x] * self.players[0][0].view(1, 1, -1)).sum(-1)
                / 2
                + self.players[0][0].view(1, -1)
                * (self.matrix[2, x] * self.players[1][0].view(1, 1, -1)).sum(-1)
                - self.players[1][0].view(1, -1)
                * (self.matrix[1, x] * self.players[1][0].view(1, 1, -1)).sum(-1)
                / 2
                + self.players[0][0].view(1, -1) * self.bias[0, x]
                - self.players[1][0].view(1, -1) * self.bias[1, x]  # + self.bias[2, x]
            )
            .sum(-1)
            .mean()
        )

        return [loss, -loss]

    def dist2opt(self):
        d = ((self.players[0][0].data.detach().cpu() - self.x_star) ** 2).sum() + (
            (self.players[1][0].data.detach().cpu() - self.y_star) ** 2
        ).sum()
        return d

    def grad_at_optimum(self):
        self.players[0][0].data = torch.tensor(self.x_star)
        self.players[1][0].data = torch.tensor(self.y_star)
        grad_norm = 0
        for x in self.sampler.iterator():
            grad = self.grad(x)
            for i in range(len(self.players)):
                for g in grad[i]:
                    grad_norm += (g ** 2).sum()
        return grad_norm

    def optimum(self):
        matrix = self.matrix.mean(1).cpu().numpy()
        matrix = np.block([[matrix[0], matrix[2]], [-matrix[2].T, matrix[1]]])
        b = np.concatenate([self.bias[0], self.bias[1]], axis=-1).mean(0)
        sol = linalg.solve(matrix, -b)
        x_star, y_star = np.split(sol, 2)
        return x_star, y_star

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.matrix = self.matrix.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self


class CommutativeQuadraticGame(QuadraticGame):
    def __init__(self, n_samples, dim, mu=(0, 0, 0), L=(1, 1, 1)):
        A, C, B = make_commutative_matrix(n_samples, dim, mu=mu, L=L)
        super().__init__(dim, n_samples, A=A, B=B, C=C)


class BilinearGame(QuadraticGame):
    def __init__(self, dim, num_samples=1, **kwargs):
        A = torch.zeros(num_samples, dim, dim)
        C = torch.zeros(num_samples, dim, dim)
        super().__init__(dim, num_samples=num_samples, A=A, C=C, **kwargs)


class L2RegressionSP(Game):
    def __init__(self, dim=50, n=10, reg=None):
        super().__init__()

        self.sampler = RandomSampler(1)

        self.dim = dim
        self.n = n
        self.b = torch.zeros(self.dim)

        self.reg = reg
        if self.reg is None:
            self.reg = 1 / n

        self.A = torch.randn(dim, dim)

        self.players = nn.ModuleList(
            [
                nn.ParameterList([nn.Parameter(torch.zeros(self.dim))]),
                nn.ParameterList([nn.Parameter(torch.zeros(self.dim))]),
            ]
        )

        self.init()

    def init(self):
        self.players[0][0].data = (
            1.0 / math.sqrt(self.dim) * torch.zeros(self.dim).normal_()
        )
        self.players[1][0].data = (
            1.0 / math.sqrt(self.dim) * torch.zeros(self.dim).normal_()
        )

    def loss(self, x=None):
        if x is None:
            x = self.sampler.sample_batch()

        loss = 0.5 * self.reg * (self.players[0][0] ** 2).sum(-1) + 1 / self.n * (
            -0.5 * (self.players[1][0] ** 2).sum(-1)
            - (self.b * self.players[1][0]).sum(-1)
            + (self.players[0][0] * self.A.mv(self.players[1][0])).sum(-1)
        )

        return [loss, -loss]

    def dist2opt(self):
        if self.x_star is None or self.y_star is None:
            self.x_star, self.y_star = (
                torch.zeros_like(self.players[0][0]),
                torch.zeros_like(self.players[1][0]),
            )
        d = ((self.players[0][0].data.detach() - self.x_star) ** 2).sum() + (
            (self.players[1][0].data.detach() - self.y_star) ** 2
        ).sum()
        return d


class RidgeRegressionSP(Game):
    def __init__(self, dim=50, num_samples=10, lambda_reg=None):
        super(RidgeRegressionSP, self).__init__()
        self.sampler = RandomSampler(num_samples)
        self.num_samples = num_samples
        self.dim = dim
        if lambda_reg is None:
            lambda_reg = 1 / num_samples
        self.lambda_reg = lambda_reg

        self.x_star = None
        self.y_star = None

        self.matrix = torch.zeros(num_samples, dim).normal_()

        self.players = self.players = nn.ModuleList(
            [
                nn.ParameterList([nn.Parameter(torch.zeros(self.dim))]),
                nn.ParameterList([nn.Parameter(torch.zeros(self.num_samples))]),
            ]
        )

    def init(self):
        self.players[0][0].data = (
            1.0 / math.sqrt(self.dim) * torch.zeros(self.dim).normal_()
        )
        self.players[1][0].data = (
            1.0 / math.sqrt(self.num_samples) * torch.zeros(self.num_samples).normal_()
        )

    def loss(self, x=None):
        if x is None:
            loss = (
                self.matrix.mv(self.players[0][0]).dot(self.players[1][0])
                / self.num_samples
                + (self.players[0][0] ** 2).sum() / 2
                - (self.players[1][0] ** 2).sum() / 2
            )
        else:
            loss = (
                self.matrix[x].mv(self.players[0][0]) * self.players[1][0][x]
                + (self.players[0][0] ** 2).sum() / (2 * self.num_samples)
                - (self.players[1][0] ** 2).sum() / (2 * self.num_samples)
            )

        return [loss, -loss]

    def dist2opt(self):
        if self.x_star is None or self.y_star is None:
            self.x_star, self.y_star = self.optimum()
        d = ((self.players[0][0] - self.x_star) ** 2).sum() + (
            (self.players[1][0] - self.y_star) ** 2
        ).sum()
        return d

    def optimum(self):
        return (
            torch.zeros_like(self.players[0][0]),
            torch.zeros_like(self.players[1][0]),
        )

