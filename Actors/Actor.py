import torch


class Actor:
    def __init__(self):
        pass

    def act(self, state_batch: torch.Tensor):
        pass

    def reset(self):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
