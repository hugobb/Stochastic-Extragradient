import math


class LRScheduler:
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, k, index=None):
        raise NotImplementedError

    def __repr__(self):
        return "lr=%.1e" % self.lr

    def __str__(self):
        return "lr=%.1e" % self.lr


class BaseLR(LRScheduler):
    def __call__(self, k, index=None):
        return self.lr


class LambdaLR(LRScheduler):
    def __init__(self, lr, func):
        self.lr = lr
        self.func = func

    def __call__(self, k, index=None):
        return self.func(k)


class DecreasingLR(LRScheduler):
    def __init__(self, lr, threshold):
        self.lr = lr
        self.threshold = threshold

    def __call__(self, k, index=None):
        if k <= math.ceil(self.threshold):
            return self.lr
        else:
            return self.lr * self.threshold * (2 * k + 1) / (2 * (k + 1) ** 2)
