import numpy as np
import pandas as pd

from Actors import ActorCritic, HardCoded
from Environments import Intersection
from Utils import EpisodeGenerator, EpisodeAnalyzer

if __name__ == '__main__':
    np_cars = np.load('Data/Elaborated/np_cars.npy')
    crossing_ped_df = pd.read_pickle('Data/Elaborated/crossing_ped_df.pkl')
    traj_df = pd.read_pickle('Data/Elaborated/traj_df.pkl')

    generate_examples = True
    ep_ex_list = [100012427, 330, 641, 1146]

    actor = None
    # actor_type = 'RL'
    actor_type = 'Human'
    # actor_type = 'HardCoded'
    rl_actor_path = 'Data/Trained_Models/RL_agent_00'

    out_path = f"Data/Results/{actor_type}"

    env = Intersection(np_cars, crossing_ped_df)

    if actor_type == 'RL':
        actor = ActorCritic(env.observation_space, env.action_space)
        actor.load(rl_actor_path)
    elif actor_type == 'Human':
        actor = None
    elif actor_type == 'HardCoded':
        actor = HardCoded()

    print('Generating episodes')
    ep_gen = EpisodeGenerator(env, traj_df)
    ep_dict = ep_gen.generate(crossing_ped_df.id, actor)
    if generate_examples:
        ep_gen.generate(ep_ex_list, actor, generate_films=True, film_path=out_path)
    print('Episodes generated')

    print('Analyzing episodes')
    ep_analyzer = EpisodeAnalyzer()
    cr_times, fig = ep_analyzer.crossing_times(ep_dict, graph=True)
    fig.savefig(f'{out_path}/crossing_time.jpg')
    min_dist, fig = ep_analyzer.ep_min_dists(ep_dict, graph=True)
    fig.savefig(f'{out_path}/min_dist.jpg')
    min_PET, fig = ep_analyzer.ep_min_PET(ep_dict, graph=True)
    fig.savefig(f'{out_path}/min_PET.jpg')
    fig = ep_analyzer.velocities_graph(ep_dict)
    fig.savefig(f'{out_path}/velocities.jpg')
    fig = ep_analyzer.traj_graph(ep_dict)
    fig.savefig(f'{out_path}/traj_graph.jpg')
