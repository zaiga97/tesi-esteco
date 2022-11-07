import os
import argparse
from datetime import datetime
import re

from Actors import ActorCritic
from Environments import make_intersection
from Algorithms import PPO, DDPG, Trainer, GAIL, AIRL
from Utils import SerializedBuffer


def run(args):
    algo = None
    env = make_intersection()
    env_test = make_intersection()
    actor = ActorCritic(env.observation_space, env.action_space)

    if args.algo_id == 'DDPG':
        algo = DDPG(
            actor_critic=actor,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            seed=args.seed
        )

    elif args.algo_id == 'PPO':
        algo = PPO(
            actor_critic=actor,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device='cpu',
            seed=args.seed
        )

    elif args.algo_id == 'GAIL':
        env.step_scale = 1.
        env_test.step_scale = 1.
        buffer_exp = SerializedBuffer(
            path='Data/Elaborated/exp_buffer',
            device='cpu'
        )

        algo = GAIL(
            actor_critic=actor,
            buffer_exp=buffer_exp,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device='cpu',
            seed=args.seed
        )

    elif args.algo_id == 'AIRL':
        env.step_scale = 1.
        env_test.step_scale = 1.
        buffer_exp = SerializedBuffer(
            path='Data/Elaborated/exp_buffer',
            device='cpu'
        )

        algo = AIRL(
            actor_critic=actor,
            buffer_exp=buffer_exp,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device='cpu',
            seed=args.seed
        )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join('logs', f'{args.algo_id}', f'seed{args.seed}-{time}')

    trainer = Trainer(
        actor=actor,
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()

    # Save the learned model
    model_dir = f'Data/Trained_Models/{args.algo_id}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    files = [file for file in os.listdir(model_dir)]
    max_v = 0
    for file in files:
        v = int(re.search('.(\d*)', file).group(1))
        max_v = v if v > max_v else max_v

    model_path = os.path.join(model_dir, f'v{max_v+1}')
    actor.save(model_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=2 * (10 ** 6))
    p.add_argument('--eval_interval', type=int, default=10 ** 4)
    p.add_argument('--algo_id', type=str, default='AIRL')
    p.add_argument('--cuda', action='store_true', default=False)
    p.add_argument('--seed', type=int, default=555)
    args = p.parse_args()
    run(args)
