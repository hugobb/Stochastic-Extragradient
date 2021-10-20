import torch
import numpy as np


class Sampler:
    def __init__(self, num_samples, seed=None):
        self.rng = np.random.default_rng(seed)
        self.num_samples = num_samples

    def sample(self, batch_size=1):
        return None

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)


class RandomSampler(Sampler):
    def __init__(self, num_samples, seed=None):
        super().__init__(num_samples, seed)
        self.index = 0

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def iterator(self):
        for i in range(self.num_samples):
            x = torch.Tensor([i]).long()
            yield x

    def sample(self, batch_size=1, return_index=False):
        x = torch.tensor(self.rng.integers(self.num_samples, size=(batch_size,))).long()
        if return_index:
            return x, x
        else:
            return x

    def sample_batch(self):
        return torch.arange(self.num_samples).long()


class DoubleLoopSampler(Sampler):
    def __init__(self, sampler):
        self.sampler = sampler
        self.num_samples = self.sampler.num_samples ** 2

    def iterator(self):
        for i in range(self.sampler.num_samples):
            for j in range(self.sampler.num_samples):
                x = (torch.Tensor([i]).long(), torch.Tensor([j]).long())
                yield x

    def sample(self, batch_size=1, return_index=False):
        x = (self.sampler.sample(batch_size), self.sampler.sample(batch_size))
        if return_index:
            i = x[0] * self.sampler.num_samples + x[1]
            return x, i
        else:
            return x
