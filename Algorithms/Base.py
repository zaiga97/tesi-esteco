from abc import ABC, abstractmethod
import numpy as np
import torch


class Algorithm(ABC):

    def __init__(self, seed):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0

    @abstractmethod
    def is_update(self):
        pass

    @abstractmethod
    def update(self, writer):
        pass
