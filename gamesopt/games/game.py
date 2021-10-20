from torch import autograd
from torch import nn
from .sampler import DoubleLoopSampler
import torch


class Game(nn.Module):
    def __init__(self):
        super(Game, self).__init__()
        self.players = None
        self.sampler = None

    @property
    def num_players(self):
        return len(self.players)

    def get_players(self):
        return self.players

    def loss(self, x=None):
        # Return a list of losses for each players
        raise NotImplementedError()

    def grad(self, x=None, players=None, return_loss=False):
        grad_all = []
        loss_all = self.loss(x)
        for loss, player in zip(loss_all, self.get_players()):
            grad = autograd.grad(
                loss, player.parameters(), retain_graph=True, create_graph=True
            )
            grad_all.append(grad)

        if return_loss:
            return grad_all, loss_all

        return grad_all

    def init(self, seed=1234):
        torch.manual_seed(seed)
        self.reset()
        self.sampler.seed(seed)

    def reset(self):
        raise NotImplementedError()

    def dist2opt(self):
        raise NotImplementedError()

    def hamiltonian(self, grad_0=None, grad_1=None):
        if grad_0 is None:
            grad_0 = self.grad()
        if grad_1 is None:
            grad_1 = self.grad()

        hamiltonian = 0
        for i in range(len(self.players)):
            for g0, g1 in zip(grad_0[i], grad_1[i]):
                hamiltonian += (g0 * g1).sum()
        hamiltonian /= 2

        return hamiltonian

    def sample(self, batch_size=1, return_index=False):
        return self.sampler.sample(batch_size, return_index)


class HamiltonianWrapper(Game):
    def __init__(self, game, mu=None, L=None):
        super(HamiltonianWrapper, self).__init__()
        self.game = game
        if self.game.sampler is not None:
            self.sampler = DoubleLoopSampler(self.game.sampler)
        self.mu = mu
        self.L = L
        self.init()
        self.players = game.players

    def hamiltonian(self):
        return self.game.hamiltonian()

    def get_players(self):
        return self.game.players

    def init(self):
        return self.game.init()

    def dist2opt(self):
        return self.game.dist2opt()

    def loss(self, x=None):
        if x is None:
            grad_0 = self.game.grad()
            grad_1 = grad_0
        else:
            grad_0 = self.game.grad(x[0])
            grad_1 = self.game.grad(x[1])

        hamiltonian = 0
        for i in range(len(self.get_players())):
            for g_0, g_1 in zip(grad_0[i], grad_1[i]):
                hamiltonian += (g_0 * g_1).sum()
        hamiltonian /= 2

        return [hamiltonian] * len(self.get_players())

