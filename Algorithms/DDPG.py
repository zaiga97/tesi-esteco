from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam

from Actors import ActorCritic
from Utils import Buffer
from .Base import Algorithm


class DDPG(Algorithm):

    def __init__(self, actor_critic: ActorCritic, state_shape, action_shape, seed, device='cpu', gamma=0.995,
                 update_every=10, buffer_size=int(1e6), polyak=0.995, lr_actor=1e-3, lr_critic=1e-3, batch_size=64,
                 start_steps=10000, act_noise=0.2):
        super().__init__(seed)

        # Rollout buffer.
        self.buffer = Buffer(
            buffer_size=buffer_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
        )

        # Actor-Critic
        self.ac = actor_critic
        self.ac_targ = deepcopy(self.ac)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr_actor)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=lr_critic)
        # Freeze target networks (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Learning parameters
        self.update_every = update_every
        self.polyak = polyak
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.act_noise = act_noise
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

    def is_update(self):
        return self.learning_steps % self.update_every == 0

    def get_action(self, state):
        action = self.ac.exploit(state)
        action += self.act_noise * np.random.randn(self.action_shape[0])
        return action

    def step(self, env, state, t, step):
        self.learning_steps += 1
        t += 1
        if step > self.start_steps:
            action = self.get_action(state)
        else:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env.max_episode_steps else done

        self.buffer.append(state, action, reward, mask, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, writer):
        batch = self.buffer.sample(self.batch_size)
        loss_q = self.update_q(batch)
        loss_pi = self.update_pi(batch)
        self.update_target()
        writer.add_scalar('loss/critic', loss_q.item(), self.learning_steps)
        writer.add_scalar('loss/actor', loss_pi.item(), self.learning_steps)

    def update_q(self, batch):
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(batch)
        loss_q.backward()
        self.q_optimizer.step()
        return loss_q

    def update_pi(self, batch):
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_optimizer.step()
        return loss_pi

    def update_target(self):
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def compute_loss_q(self, batch):
        o, a, r, d, o2 = batch
        r = r.squeeze(-1)
        d = d.squeeze(-1)
        qv = self.ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((qv - backup) ** 2).mean()
        return loss_q

    def compute_loss_pi(self, batch):
        o, _, _, _, _ = batch
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()
