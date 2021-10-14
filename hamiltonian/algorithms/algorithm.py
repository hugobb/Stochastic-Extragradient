import torch
from collections import defaultdict
from .lr_scheduler import BaseLR, LRScheduler
from collections.abc import Iterable
import json
import uuid
import os


class Algorithm:
    def __init__(
        self,
        game,
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        alternated=False,
        device=None,
        save_dir=None,
        save_freq=10000,
        save_trajectory=False,
    ):
        self.game = game

        self.lr = lr
        if not isinstance(self.lr, Iterable):
            self.lr = [self.lr] * game.num_players
        assert len(self.lr) == game.num_players

        for i, lr in enumerate(self.lr):
            if not isinstance(lr, LRScheduler):
                self.lr[i] = BaseLR(lr)

        self.momentum = momentum
        self.nesterov = nesterov
        self.alternated = alternated
        self.buf = {}
        self.game = self.game.to(device)

        self.k = 0

        self.save_freq = save_freq
        self.n_samples = 0

        self.save_trajectory = save_trajectory

        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def save(self, results, exp_id):
        if self.save_dir is not None:
            path = os.path.join(self.save_dir, "%s.json" % exp_id)
            with open(path, "w") as f:
                json.dump(results, f)

    def run(self, num_iter=100, seed=1234):
        exp_id = uuid.uuid4()
        self.game.init(seed)
        results = defaultdict(list)
        results["dist2opt"].append(self.game.dist2opt().data.cpu().item())
        results["hamiltonian"].append(self.game.hamiltonian().data.cpu().item())
        results["n_samples"].append(self.n_samples)
        if self.save_trajectory:
            results["x"].append(
                (
                    self.game.players[0][0].data.detach().clone().item(),
                    self.game.players[1][0].data.detach().clone().item(),
                )
            )
        for i in range(num_iter):
            self.update()

            results["dist2opt"].append(self.game.dist2opt().data.cpu().item())
            results["hamiltonian"].append(self.game.hamiltonian().data.cpu().item())
            results["n_samples"].append(self.n_samples)
            if self.save_trajectory:
                results["x"].append(
                    (
                        self.game.players[0][0].data.detach().clone().item(),
                        self.game.players[1][0].data.detach().clone().item(),
                    )
                )
            self.k += 1

            if i % self.save_freq == 0:
                self.save(results, exp_id)

        self.save(results, exp_id)

        return results

    def update(self):
        raise NotImplementedError

    def gradient_update(self, p, d_p, momentum=None, nesterov=None):
        momentum = self.momentum if momentum is None else momentum
        nesterov = self.nesterov if nesterov is None else nesterov
        if momentum != 0:
            if p not in self.buf:
                self.buf[p] = torch.clone(d_p).detach()
            self.buf[p].mul_(momentum).add_(d_p)
            if nesterov:
                d_p = d_p.add(self.buf[p], alpha=momentum)
            else:
                d_p = self.buf[p]

        return d_p
