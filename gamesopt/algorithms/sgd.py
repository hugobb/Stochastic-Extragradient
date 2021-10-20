from .algorithm import Algorithm
import copy
import torch


class SGD(Algorithm):
    # The recommended learning rate is 1/L
    def __init__(self, game, full_batch=False, *args, **kwargs):
        super(SGD, self).__init__(game, *args, **kwargs)
        self.full_batch = full_batch

    def grad(self):
        if self.full_batch:
            x = None
        else:
            x = self.game.sample()

        grad = self.game.grad(x)
        return grad

    def update(self):
        if not self.alternated:
            grad = self.grad()
        for i, player in enumerate(self.game.get_players()):
            if self.alternated:
                grad = self.grad()
            for p, g in zip(player.parameters(), grad[i]):
                d_p = self.gradient_update(p, g)
                p.data.add_(d_p, alpha=-self.lr[i](self.k))


class Nesterov(SGD):
    # The recommended learning rate is 1/L and momentum is 1-q/(1+q) where q=sqrt(mu/L)
    def __init__(self, game, full_batch=False, *args, **kwargs):
        super(Nesterov, self).__init__(game, full_batch, *args, **kwargs)
        self.buf = {}
        self.t = 0

    def update(self):
        momentum_coeff = self.momentum

        if not self.alternated:
            grad = self.grad()
        for i, player in enumerate(self.game.get_players()):
            if self.alternated:
                grad = self.grad()
            for p, g in zip(player.parameters(), grad[i]):
                x = p.data - self.lr[i] * g
                if p not in self.buf:
                    self.buf[p] = torch.clone(p).detach()
                d = x - self.buf[p]
                p.data = x + momentum_coeff * d
                self.buf[p] = torch.clone(x).detach()

        self.t += 1


class SVRG(Algorithm):
    def __init__(self, game, prob=None, *args, **kwargs):
        super(SVRG, self).__init__(game, *args, **kwargs)
        if prob is None:
            prob = self.game.sampler.num_samples
        self.prob = prob

        self.snapshot = copy.deepcopy(self.game)
        self.full_grad = self.snapshot.grad()

    def update(self):
        if not self.alternated:
            x = self.game.sample()
            grad_snapshot = self.snapshot.grad(x)
            grad = self.game.grad(x)
        for i, player in enumerate(self.game.get_players()):
            if self.alternated:
                x = self.game.sample()
                grad_snapshot = self.snapshot.grad(x)
                grad = self.game.grad(x)
            for p, g, gs, bg in zip(
                player.parameters(), grad[i], grad_snapshot[i], self.full_grad[i]
            ):
                d_p = g - gs + bg
                d_p = self.gradient_update(p, d_p)
                p.data.add_(d_p, alpha=-self.lr)

        coin = torch.rand(1)
        if coin < self.prob:
            self.snapshot.load_state_dict(self.game.state_dict())
            self.full_grad = self.snapshot.grad()


class SAGA(Algorithm):
    def __init__(self, *args, **kwargs):
        super(SAGA, self).__init__(*args, **kwargs)

        self.grad_sum = {}
        self.grad_list = []
        for x in self.game.sampler.iterator():
            grad = self.game.grad(x)
            for i, player in enumerate(self.game.get_players()):
                for p, g in zip(player.parameters(), grad[i]):
                    if p not in self.grad_sum:
                        self.grad_sum[p] = torch.zeros_like(p)
                    self.grad_sum[p] += g

            self.grad_list.append(grad)

    def update(self):
        x, index = self.game.sample(return_index=True)
        grad = self.game.grad(x)

        for i, player in enumerate(self.game.get_players()):
            for p, g, g_list in zip(
                player.parameters(), grad[i], self.grad_list[index][i]
            ):
                d_p = g - g_list + self.grad_sum[p] / self.game.sampler.num_samples
                d_p = self.gradient_update(p, d_p)
                p.data.add_(d_p, alpha=-self.lr)
                self.grad_sum[p] += g - g_list
                g_list.data = g.data.clone()


class Katyusha(Algorithm):
    # Based on https://arxiv.org/pdf/1901.08689.pdf
    def __init__(self, game, theta_1, theta_2, prob, mu, L):
        super(Katyusha, self).__init__(game)
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.lr = theta_2 / ((1 + theta_2) * theta_1)
        self.sigma = mu / L
        self.mu = mu
        self.L = L
        self.prob = prob

        self.omega = copy.deepcopy(self.game)
        self.grad_omega = self.omega.grad()

        self.z = copy.deepcopy(self.game.get_players())
        self.y = copy.deepcopy(self.game.get_players())

    def update(self):
        i = self.game.sample()
        grad_omega_i = self.omega.grad(i)

        for player_num, (player, player_omega) in enumerate(
            zip(self.game.get_players(), self.omega.get_players())
        ):
            for x, z, omega, y in zip(
                player.parameters(),
                self.z[player_num].parameters(),
                player_omega.parameters(),
                self.y[player_num].parameters(),
            ):
                x.data = (
                    self.theta_1 * z
                    + self.theta_2 * omega
                    + (1 - self.theta_1 - self.theta_2) * y
                )

        grad_i = self.game.grad(i)

        for player_num, player in enumerate(self.game.get_players()):
            for x, y, z, g_i, g_omega_i, g_omega in zip(
                player.parameters(),
                self.y[player_num].parameters(),
                self.z[player_num].parameters(),
                grad_i[player_num],
                grad_omega_i[player_num],
                self.grad_omega[player_num],
            ):

                g = g_i - g_omega_i + g_omega
                z_new = (
                    1
                    / (1 + self.lr * self.sigma)
                    * (self.lr * self.sigma * x + z - self.lr / self.L * g)
                )
                y.data.copy_(x).add_(z_new - z, alpha=self.theta_1)
                z.data.copy_(z_new)

        coin = torch.rand(1)
        if coin < self.prob:
            self.omega.get_players().load_state_dict(self.y.state_dict())
            self.grad_omega = self.omega.grad()

