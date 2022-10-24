import torch.nn.functional as F
from torch.optim import Adam

from Actors import ActorCritic
from ._PPO import PPO


class GAIL(PPO):

    def __init__(self, buffer_exp, disc, actor_critic: ActorCritic, state_shape, action_shape, device, batch_size=64, lr_disc = 3e-4, epoch_disc: int = 10,
                 seed: int = 0, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4, max_ep_len=200, lr_critic=3e-4, epoch_ppo=10,
                 clip_eps=0.2, lambd=0.97, coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(actor_critic, state_shape, action_shape, device, seed, gamma, rollout_length, mix_buffer,
                         lr_actor, max_ep_len, lr_critic, epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm)

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = disc.to(device)

        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self):

        for _ in range(self.epoch_disc):
            # Samples from current policy's trajectories.
            states, actions = self.buffer.sample(self.batch_size)[:2]
            # Samples from expert's demonstrations.
            states_exp, actions_exp = self.buffer_exp.sample(self.batch_size)[:2]
            # Update discriminator.
            self.update_disc(states, actions, states_exp, actions_exp)

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # Calculate rewards.
        rewards = self.disc.calculate_reward(states, actions)

        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states)

    def update_disc(self, states, actions, states_exp, actions_exp):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, actions)
        logits_exp = self.disc(states_exp, actions_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()
