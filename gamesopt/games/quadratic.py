from .game import Game
from .sampler import RandomSampler
import torch
import math
import torch.nn as nn


def make_random_matrix(num_samples, dim, mu=0., L=1., max_im=1.):
    if isinstance(L, float):
        L_min = L
        L_max = L
    elif len(L) == 2:
        L_min = L[0]
        L_max = L[1]
    else:
        raise ValueError()

    matrix = torch.randn(num_samples, dim, dim)
    _, matrix = torch.linalg.eig(matrix)
    
    L_i = torch.rand(num_samples, 1)
    L_i = (L_i - L_i.min()) / (L_i.max() - L_i.min())
    L_i = L_min + L_i * (L_max - L_min)

    real_part = torch.rand(num_samples, dim)
    real_part = (real_part - real_part.min()) / (real_part.max() - real_part.min())
    real_part = mu + real_part * (L_i - mu)
    
    im_part = torch.rand(num_samples, dim)
    im_part = (im_part - im_part.min()) / (im_part.max() - im_part.min())
    im_part = (2*im_part - 1)*max_im

    eigs = torch.complex(real_part, im_part)

    matrix = torch.matmul(matrix, torch.matmul(eigs.diag_embed(), matrix.inverse())).real
    matrix[:, :dim, :dim] = 0.5*(matrix[:, :dim, :dim].transpose(-1, -2) + matrix[:, :dim, :dim])
    matrix[:, dim:, dim:] = 0.5*(matrix[:, dim:, dim:].transpose(-1, -2) + matrix[:, dim:, dim:])
    
    s = torch.linalg.eigvals(matrix)
    print(s.real.max(dim=-1)[0].min(), s.real.max(dim=-1)[0].max(), s.real.min(), s.imag.max(), abs(s.imag).min())
    return matrix


class QuadraticGame(Game):
    def __init__(
        self,
        dim,
        num_samples=1,
        bias=False,
        mu=0.,
        L=1.,
        max_im=1.,
        init_func=None
    ):
        super().__init__()

        self.dim = dim
        self.num_samples = num_samples
        self.sampler = RandomSampler(num_samples)

        self.matrix = make_random_matrix(num_samples, 2*dim, mu=mu, L=L, max_im=max_im)

        self.bias = torch.zeros(2, num_samples, dim)
        if bias:
            self.bias = 1.0 / math.sqrt(self.dim) * self.bias.normal_() / 10

        self.x_star, self.y_star = self.optimum()

        self.players = nn.ModuleList(
            [
                nn.ParameterList([nn.Parameter(torch.zeros(self.dim))]),
                nn.ParameterList([nn.Parameter(torch.zeros(self.dim))]),
            ]
        )

        """
        J_ij = torch.zeros(num_samples, num_samples, 2 * dim, 2 * dim)
        for i in range(num_samples):
            for j in range(num_samples):
                J_ij[i, j] = (
                    torch.mm(self.matrix[i].transpose(-1, -2), self.matrix[j])
                    + torch.mm(self.matrix[j].transpose(-1, -2), self.matrix[i])
                ) / 2
        J_ij = J_ij.view(-1, 2 * dim, 2 * dim)

        s = torch.linalg.eigvalsh(J_ij.cpu())
        s_mean = torch.svd(self.matrix.mean(0))[1]
        e = torch.linalg.eigvals(self.matrix)
        e_mean = torch.linalg.eigvals(self.matrix.mean(0))

        self.ell_xi = float((1 / ((1 / e).real.min(1))).max())
        self.mu = float(e_mean.real.min())
        self.cL_H = float(s.max().item())
        self.mu_H = float(s_mean.min().item()) ** 2
        """
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
        A = self.matrix[:, :self.dim, :self.dim]
        B = self.matrix[:, :self.dim, self.dim:]
        C = self.matrix[:, self.dim:, :self.dim]
        D = self.matrix[:, self.dim:, self.dim:]

        if x is None:
            x = self.sampler.sample_batch()

        loss_1 = (
            (
                self.players[0][0].view(1, -1)
                * (A[x] * self.players[0][0].view(1, 1, -1)).sum(-1)
                / 2
                + self.players[0][0].view(1, -1)
                * (B[x] * self.players[1][0].view(1, 1, -1)).sum(-1)
                + self.players[0][0].view(1, -1) * self.bias[0, x]
            ).sum(-1)
            ).mean()

        loss_2 = (
            (
                self.players[1][0].view(1, -1)
                * (D[x] * self.players[1][0].view(1, 1, -1)).sum(-1)
                / 2
                + self.players[1][0].view(1, -1)
                * (C[x] * self.players[0][0].view(1, 1, -1)).sum(-1)
                + self.players[1][0].view(1, -1) * self.bias[1, x]
            )
            .sum(-1)
            .mean()
        )

        return [loss_1, loss_2]

    def dist2opt(self):
        d = ((self.players[0][0].data - self.x_star) ** 2).sum() + (
            (self.players[1][0].data - self.y_star) ** 2
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
        b = torch.cat([self.bias[0], self.bias[1]], dim=-1).mean(0)
        sol = torch.linalg.solve(self.matrix.mean(0), -b)
        x_star, y_star = torch.split(sol, 2)
        return x_star, y_star

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.matrix = self.matrix.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self