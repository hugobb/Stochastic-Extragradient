from .algorithm import Algorithm
from hamiltonian.games.sampler import DoubleLoopSampler
import torch.autograd as autograd
from .lr_scheduler import BaseLR, LRScheduler
import copy
import torch


class ConsensusOptimization(Algorithm):
    ### The recommended learning rate is 1/L
    def __init__(
        self, game, lr_H=None, batch_size=1, full_batch=False, *args, **kwargs
    ):
        super().__init__(game, *args, **kwargs)
        if lr_H is None:
            lr_H = self.lr
        self.lr_H = lr_H
        if not isinstance(lr_H, LRScheduler):
            self.lr_H = BaseLR(self.lr_H)

        self.batch_size = batch_size
        self.full_batch = full_batch
        self.sampler = DoubleLoopSampler(game.sampler)

    def update(self):
        if self.full_batch:
            grad = self.game.grad(None)
            grad_1, grad_2 = grad, grad
            self.n_samples += self.game.sampler.num_samples
        else:
            x_1, x_2 = self.sampler.sample(self.batch_size)
            grad_1, grad_2 = self.game.grad(x_1), self.game.grad(x_2)
            self.n_samples += 2 * self.batch_size

        hamiltonian = self.game.hamiltonian(grad_1, grad_2)
        grad_H = []
        for player in self.game.get_players():
            _g = autograd.grad(hamiltonian, player.parameters(), retain_graph=True)
            grad_H.append(_g)

        for i, player in enumerate(self.game.get_players()):
            for p, g1, g2, gH in zip(
                player.parameters(), grad_1[i], grad_2[i], grad_H[i]
            ):
                p.data += -self.lr[i](self.k) / 2 * (g1 + g2) - self.lr_H(self.k) * gH


class SVRCO(ConsensusOptimization):
    def __init__(self, game, prob=None, *args, **kwargs):
        super().__init__(game, *args, **kwargs)
        if prob is None:
            prob = self.game.sampler.num_samples
        self.prob = prob

        self.snapshot = copy.deepcopy(self.game)
        self.full_grad = self.snapshot.grad()
        hamiltonian = self.game.hamiltonian()
        self.full_grad_H = []
        for player in self.game.get_players():
            _g = autograd.grad(hamiltonian, player.parameters(), retain_graph=True)
            self.full_grad_H.append(_g)
        self.n_samples += self.game.sampler.num_samples

    def update(self):
        if self.full_batch:
            grad = self.game.grad()
            grad_snapshot = self.snapshot.grad()
            grad_1, grad_2 = grad, grad
            grad_snapshot_1, grad_snapshot_2 = grad_snapshot, grad_snapshot
            self.n_samples += self.game.sampler.num_samples
        else:
            x_1, x_2 = self.sampler.sample(self.batch_size)
            grad_1, grad_2 = self.game.grad(x_1), self.game.grad(x_2)
            grad_snapshot_1, grad_snapshot_2 = (
                self.snapshot.grad(x_1),
                self.snapshot.grad(x_2),
            )
            self.n_samples += 2 * self.batch_size

        hamiltonian = self.game.hamiltonian(grad_1, grad_2)
        grad_H = []
        for player in self.game.get_players():
            _g = autograd.grad(hamiltonian, player.parameters(), retain_graph=True)
            grad_H.append(_g)

        hamiltonian = self.snapshot.hamiltonian(grad_snapshot_1, grad_snapshot_2)
        snapshot_grad_H = []
        for player in self.snapshot.get_players():
            _g = autograd.grad(hamiltonian, player.parameters(), retain_graph=True)
            snapshot_grad_H.append(_g)

        for i, player in enumerate(self.game.get_players()):
            for p, g1, g2, gs1, gs2, bg, g_H, gs_H, bg_H in zip(
                player.parameters(),
                grad_1[i],
                grad_2[i],
                grad_snapshot_1[i],
                grad_snapshot_2[i],
                self.full_grad[i],
                grad_H[i],
                snapshot_grad_H[i],
                self.full_grad_H[i],
            ):
                g_1 = (g1 + g2) / 2 - (gs1 + gs2) / 2 + bg
                g_2 = g_H - gs_H + bg_H
                p.data += -self.lr[i](self.k) * g_1 - self.lr_H(self.k) * g_2

        coin = torch.rand(1)
        if coin < self.prob:
            self.snapshot.load_state_dict(self.game.state_dict())
            self.full_grad = self.snapshot.grad()
            hamiltonian = self.game.hamiltonian()
            self.full_grad_H = []
            for player in self.game.get_players():
                _g = autograd.grad(hamiltonian, player.parameters(), retain_graph=True)
                self.full_grad_H.append(_g)
            self.n_samples += self.game.sampler.num_samples
