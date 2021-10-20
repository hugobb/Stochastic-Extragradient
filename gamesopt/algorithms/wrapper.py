from .algorithm import Algorithm


class OptimizerWrapper(Algorithm):
    def __init__(self, game, optimizer, full_batch=False, return_loss=False):
        self.game = game
        self.optimizer = optimizer
        self.full_batch = full_batch

        self.return_loss = return_loss

    def update(self):
        if self.full_batch:
            x = None
        else:
            x = self.game.sample()

        if self.return_loss:
            grad, loss = self.game.grad(x, return_loss=self.return_loss)
        else:
            grad = self.game.grad(x, return_loss=self.return_loss)

        for i, player in enumerate(self.game.get_players()):
            for p, g in zip(player.parameters(), grad[i]):
                p.grad = g

        if self.return_loss:
            self.optimizer.step(loss=loss[0])  # TODO: handle multiple loss.
        else:
            self.optimizer.step()
