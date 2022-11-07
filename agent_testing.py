import argparse
import os

import pandas as pd

from Actors import HardCoded, ActorCritic, Human
from Environments import make_intersection
from Utils import EpisodeGenerator, EpisodeAnalyzer


def test(args):
    crossing_ped_df = pd.read_pickle('Data/Elaborated/crossing_ped_df.pkl')
    traj_df = pd.read_pickle('Data/Elaborated/traj_df.pkl')

    generate_examples = True
    ep_ex_list = [100012427, 330, 641, 1146]

    version = args.actor_v
    actor_type = args.actor_type

    actor_path = f'Data/Trained_Models/{actor_type}/v{version}'
    out_path = f"Data/Results/{actor_type}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    env = make_intersection()
    #env.step_scale = 1.
    #env.flip_env()

    if actor_type == 'Human':
        env.flip_env()
        actor = Human(traj_df, scale=env.step_scale)
    elif actor_type == 'HardCoded':
        actor = HardCoded(scale=env.step_scale)
    else:
        actor = ActorCritic(env.observation_space, env.action_space)
        print(f'Loading actor from: {actor_path}')
        actor.load(actor_path)

    print('Generating episodes')
    ep_gen = EpisodeGenerator(env, traj_df)
    ep_dict = ep_gen.generate(actor, crossing_ped_df.id)
    if generate_examples:
        ep_gen.generate(actor, ep_ex_list, generate_films=True, film_path=out_path)
    print('Episodes generated')

    print('Analyzing episodes')
    ep_analyzer = EpisodeAnalyzer()
    cr_times, fig = ep_analyzer.crossing_times(ep_dict, graph=True)
    cr_times.to_csv(f'{out_path}/crossing_time.csv')
    fig.savefig(f'{out_path}/crossing_time.jpg')

    min_dist, fig = ep_analyzer.ep_min_dists(ep_dict, graph=True)
    min_dist.to_csv(f'{out_path}/min_dist.csv')
    fig.savefig(f'{out_path}/min_dist.jpg')

    min_PET, fig = ep_analyzer.ep_min_PET(ep_dict, graph=True)
    min_PET.to_csv(f'{out_path}/min_PET.csv')
    fig.savefig(f'{out_path}/min_PET.jpg')

    velocities, fig = ep_analyzer.ep_velocities(ep_dict, graph=True)
    velocities.to_csv(f'{out_path}/velocities.csv')
    fig.savefig(f'{out_path}/velocities.jpg')

    fig = ep_analyzer.traj_graph(ep_dict)
    fig.savefig(f'{out_path}/traj_graph.jpg')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--actor_type', type=str, default='DDPG')
    p.add_argument('--actor_v', type=int, default=1)
    args = p.parse_args()
    test(args)
