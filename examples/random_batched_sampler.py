import torch

from torch.utils.data import RandomSampler

class RandomBatchedSampler(RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None, batch_size=1):
        super(RandomBatchedSampler, self).__init__(data_source, replacement=replacement, num_samples=num_samples)
        self.batch_size = batch_size

    def __iter__(self):
        n = len(self.data_source)
        b = self.batch_size
        return iter(torch.cat([x * b + torch.arange(0, b) for x in torch.randperm(n // b)]).tolist())
