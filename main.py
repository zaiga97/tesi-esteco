import numpy as np
import pandas as pd
import torch

from Actors import HardCoded, ActorCritic
from Environments import Intersection
from Utils import FilmMaker, EpisodeGenerator, EpisodeAnalyzer

if __name__ == '__main__':
    np_cars = np.load('Data/Elaborated/np_cars.npy')
    crossing_ped_df = pd.read_pickle('Data/Elaborated/crossing_ped_df.pkl')
    traj_df = pd.read_pickle('Data/Elaborated/traj_df.pkl')

    env = Intersection(np_cars, crossing_ped_df)
    actor = HardCoded()
    actor = ActorCritic(env.observation_space, env.action_space)
    actor.load('Data/Trained_Models/RL_agent_final')

    ep_gen = EpisodeGenerator(env, traj_df)
    ep_dict = ep_gen.generate(crossing_ped_df.id)
    ep_an = EpisodeAnalyzer()
    fig = ep_an.traj_graph(ep_dict)
    fig.savefig('Traj')




