import torch
import torch.nn as nn

from Actors import Actor


class PI(nn.Module):
    def __init__(self, obs_size: int, action_size: int, action_max: torch.Tensor, hidden_dim: int = 128):
        super(PI, self).__init__()
        self.action_max = action_max
        self.in_block = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.action_dir = nn.Sequential(
            nn.Linear(hidden_dim, action_size)
        )

        self.action_mag = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        x = self.in_block(obs)
        action_dir = self.action_dir(x)
        action_dir = torch.nn.functional.normalize(action_dir, dim=-1)
        action_mag = self.action_mag(x) * self.action_max
        return action_dir * action_mag


class Q(nn.Module):
    def __init__(self, obs_size: int, action_size: int, hidden_dim: int = 256):
        super(Q, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_size + action_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-1)
                nn.init.zeros_(m.bias)

    def forward(self, obs, acts):
        return torch.squeeze(self.q(torch.cat([obs, acts], dim=-1)), -1)


class ActorCritic(nn.Module, Actor):
    def __init__(self, obs_space, action_space):
        super(ActorCritic, self).__init__()
        obs_size = obs_space.shape[0]
        action_size = action_space.shape[0]
        action_max = action_space.high[0]
        self.pi = PI(obs_size, action_size, action_max)
        self.q = Q(obs_size, action_size)

    def act(self, obs):
        with torch.no_grad():
            action = self.pi(obs).numpy()
            return action

    def reset(self):
        pass

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
