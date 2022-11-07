import torch
from torch import nn


class PI(nn.Module):
    def __init__(self, obs_size: int, action_size: int, hidden_dim: int = 64):
        super().__init__()
        self.in_block = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.action_dir = nn.Linear(hidden_dim, action_size)
        self.action_mag = nn.Linear(hidden_dim, 1)

        self.log_stds = nn.Parameter(-0.05 * torch.ones(1, action_size))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-2)
                nn.init.zeros_(m.bias)

    def net(self, obs):
        x = self.in_block(obs)
        action_dir = self.action_dir(x)
        action_dir = torch.nn.functional.normalize(action_dir, dim=-1)
        action_mag = self.action_mag(x)
        return action_dir * action_mag

    def forward(self, obs):
        return torch.tanh(self.net(obs))


class Q(nn.Module):
    def __init__(self, obs_size: int, action_size: int, hidden_dim: int = 256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_size + action_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-2)
                nn.init.zeros_(m.bias)

    def forward(self, obs, acts):
        return torch.squeeze(self.q(torch.cat([obs, acts], dim=-1)), -1)


class V(nn.Module):
    def __init__(self, obs_size: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-2)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        return self.net(obs)
