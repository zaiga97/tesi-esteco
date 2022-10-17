from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam

from Actors import ActorCritic
from Utils import ReplayBuffer


def ddpg(env_gen_fn, actor_critic: ActorCritic, seed=0, steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=64, start_steps=10000, update_after=1000, update_every=50,
         act_noise=0.1, num_test_episodes=10, max_ep_len=1000):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_gen_fn(), env_gen_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set up function for computing DDPG Q-loss
    # noinspection PyShadowingNames
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        return loss_q

    # Set up function for computing DDPG pi loss
    # noinspection PyShadowingNames
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr, weight_decay=1e-3)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    # noinspection PyShadowingNames
    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network, so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network, so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    # noinspection PyShadowingNames
    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    # noinspection PyShadowingNames
    def test_agent():
        tot_step, tot_ret = 0, 0
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            tot_step += ep_len
            tot_ret += ep_ret
        return tot_step/num_test_episodes, tot_ret/num_test_episodes

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            print(f"Epoch: {(t + 1) / steps_per_epoch}")
            # Test the performance of the deterministic version of the agent.
            avg_len, avg_ret = test_agent()
            print(f"Ep_len: {avg_len}, ep_ret: {avg_ret}")
