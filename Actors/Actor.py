import torch


class Actor:
    def __init__(self):
        super(Actor, self).__init__()

    def exploit(self, state_batch: torch.Tensor):
        raise NotImplementedError

    def explore(self, state_batch: torch.Tensor):
        raise NotImplementedError

    def reset(self, *args):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
