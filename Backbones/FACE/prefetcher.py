import torch
"""

"""
import time
class data_prefetcher():
    def __init__(self, loader):
        st = time.time()
        self.loader = iter(loader)

        self.origin_loader = iter(loader)
        # print('Generate loader took', time.time() - st)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self.next_batch.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

import random
class data_buffer_kd():
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.buffer = []

    def add(self, item):
        if not isinstance(item, torch.Tensor):
            raise TypeError("Only tensors can be added to the buffer")
        if len(self.buffer) >= self.maxsize:
            self.buffer.pop(0)  # Remove the oldest item if buffer is full
        self.buffer.append(item)

    def add_n(self, items):
        for i in items:
            self.add(i)

    def size(self):
        return len(self.buffer)

    def random_sample(self, n):
        if n >= len(self.buffer):
            return torch.stack(self.buffer)
        return torch.stack(random.sample(self.buffer, n))