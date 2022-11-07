import torch
from torch import nn
from torch.optim import Adam

from Actors import ActorCritic
from .Base import Algorithm
from Utils import RolloutBuffer


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std(dim=0) + 1e-4)


class PPO(Algorithm):

    def __init__(self, actor_critic: ActorCritic, state_shape, action_shape, seed, device='cpu', gamma=0.995,
                 rollout_length=4096, lr_actor=3e-4, lr_critic=1e-3, epoch_ppo=20, clip_eps=0.2,
                 lambd=0.3, coef_ent=0., max_grad_norm=5.0):
        super().__init__(seed)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
        )

        # Actor.
        self.ac = actor_critic

        self.optim_actor = Adam(self.ac.pi.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.ac.v.parameters(), lr=lr_critic)

        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.learning_steps_ppo = 0

    def is_update(self):
        return self.learning_steps % self.rollout_length == 0

    def step(self, env, state, t, step):
        self.learning_steps += 1
        t += 1

        action, log_pi = self.ac.explore(state)
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env.max_episode_steps else done

        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self,  writer):
        states, actions, rewards, dones, log_pis, next_states = self.buffer.get()
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states, writer):
        with torch.no_grad():
            values = self.ac.v(states)
            next_values = self.ac.v(next_states)

        targets, gaes = calculate_gae(values, rewards, dones, next_values, self.gamma, self.lambd)
        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_actor(states, actions, log_pis, gaes, writer)
            self.update_critic(states, targets, writer)

    def update_critic(self, states, targets, writer):
        loss_critic = (self.ac.v(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.ac.v.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar('info/v', self.ac.v(states).mean(), self.learning_steps)
            writer.add_scalar('info/targ', targets.mean(), self.learning_steps)
            writer.add_scalar('loss/critic', loss_critic.item(), self.learning_steps)
            writer.add_scalar('info/log_stds', self.ac.pi.log_stds.mean(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.ac.evaluate_log_pi(states, actions)
        entropy = self.ac.evaluate_entropy()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.ac.pi.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar('loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar('info/entropy', entropy.item(), self.learning_steps)
