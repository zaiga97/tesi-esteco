import torch
import torch.nn.functional as F
from torch.optim import Adam

from .PPO import PPO
from .Utils import AIRLDiscrim


class AIRL(PPO):

    def __init__(self, actor_critic, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=4096, batch_size=256, lr_actor=3e-4, lr_critic=1e-3, lr_disc=3e-4,
                 epoch_ppo=20, epoch_disc=10, clip_eps=0.2, lambd=.8, coef_ent=0., max_grad_norm=5.0):
        super().__init__(
            actor_critic, state_shape, action_shape, seed, device, gamma, rollout_length, lr_actor, lr_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            gamma=gamma
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer):

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, actions, _, dones, log_pis, next_states = self.buffer.sample(self.batch_size)
            # Samples from expert's demonstrations.
            states_exp, actions_exp, _, dones_exp, next_states_exp = self.buffer_exp.sample_with_noise(self.batch_size)
            dones_exp.unsqueeze_(-1)
            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                log_pis_exp = self.ac.evaluate_log_pi(states_exp, actions_exp)
            # Update discriminator.
            self.update_disc(states, actions, dones, log_pis, next_states, states_exp, actions_exp, dones_exp,
                             log_pis_exp, next_states_exp, writer)

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # Calculate rewards.
        rewards = self.disc.calculate_reward(states, actions, dones, log_pis, next_states)

        # Update PPO using estimated rewards.
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc(self, states, actions, dones, log_pis, next_states,
                    states_exp, actions_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, actions, dones, log_pis, next_states)
        logits_exp = self.disc(states_exp, actions_exp, dones_exp, log_pis_exp, next_states_exp)

        loss_pi = F.binary_cross_entropy_with_logits(logits_pi, torch.zeros(logits_pi.size()))
        loss_exp = F.binary_cross_entropy_with_logits(logits_exp, torch.ones(logits_exp.size()))
        loss_disc = loss_pi + loss_exp
        loss_tot = loss_disc

        self.optim_disc.zero_grad()
        loss_tot.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar('loss/disc', loss_disc.item(), self.learning_steps)
            writer.add_scalar('loss/disc', loss_tot.item(), self.learning_steps)
            # writer.add_scalar('loss/pen', loss_penalty.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
