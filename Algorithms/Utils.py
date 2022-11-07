import torch
from torch import nn, autograd


class GAILDiscrim(nn.Module):

    def __init__(self, obs_size, action_size, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size + action_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.zeros_(m.bias)

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))

    def calculate_reward(self, states, actions):
        with torch.no_grad():
            d = self.forward(states, actions)
            s = torch.sigmoid(d)
            r = torch.log(s + 1e-2) - torch.log((1-s) + 1e-2)
            return r

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.net(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, gamma, hidden_dim=64):
        super().__init__()
        obs_size = state_shape[0]
        action_size = action_shape[0]

        self.r = nn.Sequential(
            nn.Linear(obs_size + action_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.v = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.gamma = gamma

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-2)
                nn.init.zeros_(m.bias)

    def f(self, states, actions, dones, next_states):
        rw = self.r(torch.cat([states, actions], 1))
        vs = self.v(states)
        next_vs = self.v(next_states)
        return rw + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, actions, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, actions, dones, next_states) - log_pis

    def calculate_reward(self, states, actions, dones, log_pis, next_states):
        with torch.no_grad():
            d = self.forward(states, actions, dones, log_pis, next_states)
            s = torch.sigmoid(d)
            r = torch.log(s + 1e-1) - torch.log((1 - s) + 1e-1)
            return r
