from .algorithm import Algorithm
from .lr_scheduler import BaseLR, LRScheduler
import torch


class SEG(Algorithm):
    # The recommended learning rate is 1/L
    def __init__(self, game, lr_e=None, same_sample=False, full_batch=False, *args, **kwargs):
        super().__init__(game, *args, **kwargs)
        self.buf = {}
        self.same_sample = same_sample
        self.full_batch = full_batch

        self.lr_e = lr_e
        if self.lr_e is None:
            self.lr_e = self.lr
        elif isinstance(self.lr_e, float):
            self.lr_e = (BaseLR(self.lr_e),) * game.num_players
        elif isinstance(self.lr_e, LRScheduler):
            self.lr_e = (self.lr_e,) * game.num_players

        assert len(self.lr_e) == game.num_players

    def grad(self, x):
        grad = self.game.grad(x)
        return grad

    def sample(self):
        if self.full_batch:
            x = None
        else:
            x = self.game.sample(return_index=True)
        return x

    def update(self):
        x, index = self.sample()
        grad = self.grad(x)
        # Extrapolation step
        for i, player in enumerate(self.game.get_players()):
            for p, g in zip(player.parameters(), grad[i]):
                d_p = self.gradient_update(p, g)
                self.buf[p] = torch.clone(p).detach()
                p.data -= d_p*self.lr_e[i](self.k, index)

        if not self.same_sample:
            x, index = self.sample()
        grad = self.grad(x)

        # Update step
        for i, player in enumerate(self.game.get_players()):
            for p, g in zip(player.parameters(), grad[i]):
                d_p = self.gradient_update(p, g)
                p.data = self.buf[p] - self.lr[i](self.k, index) * d_p