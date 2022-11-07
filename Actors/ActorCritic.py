import math

import torch
import torch.nn as nn

from Actors import Actor
from .Networks import PI, Q, V


class ActorCritic(nn.Module, Actor):
    def __init__(self, obs_space, action_space):
        super().__init__()
        obs_size = obs_space.shape[0]
        action_size = action_space.shape[0]
        self.pi = PI(obs_size, action_size)
        self.v = V(obs_size)
        self.q = Q(obs_size, action_size)

    @staticmethod
    def calculate_log_pi(noise):
        return -0.5 * noise.pow(2).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi)

    def explore(self, obs):
        obs = torch.tensor(obs, dtype=torch.float)
        with torch.no_grad():
            mean = self.pi.net(obs.unsqueeze_(0))
            noise = torch.randn_like(mean)
            action = torch.tanh(mean + noise * self.pi.log_stds.exp())
            log_pi = self.calculate_log_pi(noise)
            return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, obs):
        obs = torch.tensor(obs, dtype=torch.float)
        with torch.no_grad():
            return self.pi(obs.unsqueeze_(0)).cpu().numpy()[0]

    def evaluate_log_pi(self, state, action):
        mean = self.pi.net(state)
        noise = (torch.atanh(action) - mean) / (self.pi.log_stds.exp() + 1e-6)
        return self.calculate_log_pi(noise)

    def evaluate_entropy(self):
        return (0.5 * torch.log(2 * math.pi * (self.pi.log_stds.exp() ** 2)) + 0.5).mean()

    def reset(self, *args):
        pass

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
