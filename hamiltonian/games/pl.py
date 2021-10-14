from .game import Game
from torch import nn
import torch
import math
from .sampler import RandomSampler


class PL2dGame(Game):
    def __init__(self):
        super().__init__()
        self.sampler = RandomSampler(1)

        self.players = nn.ModuleList(
            [
                nn.ParameterList([nn.Parameter(torch.zeros(1))]),
                nn.ParameterList([nn.Parameter(torch.zeros(1))]),
            ]
        )

        self.init()

    def init(self):
        self.players[0][0].data = torch.ones(1) * 10
        self.players[1][0].data = torch.ones(1) * 10

    def loss(self, x=None):
        loss = (
            self.players[0][0] ** 2
            + 3
            * torch.sin(self.players[0][0]) ** 2
            * torch.sin(self.players[1][0]) ** 2
            - 4 * self.players[1][0] ** 2
            - 10 * torch.sin(self.players[1][0]) ** 2
        ).sum()
        return [loss, -loss]

    def dist2opt(self):
        return (self.players[0][0] ** 2 + self.players[1][0] ** 2).sum()


class RobustLeastSquares(Game):
    def __init__(self, A, y_0, lambda_reg=3, M=None):
        super(RobustLeastSquares, self).__init__()

        self.num_samples, self.dim = A.size()
        self.sampler = RandomSampler(self.num_samples)

        self.A = A
        self.y_0 = y_0
        self.lambda_reg = lambda_reg

        self.players = nn.ModuleList(
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
        loss = (
            self.A.mv(self.players[0][0]) - self.players[1][0]
        ) ** 2 - self.lambda_reg * (self.players[1][0] - self.y_0) ** 2
        if x is None:
            loss = loss.mean()
        else:
            x = x.view(-1)
            loss = loss[x]

        return [loss, -loss]

    def dist2opt(self):
        return torch.zeros(1)
