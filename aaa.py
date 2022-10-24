import numpy as np
import pandas as pd

from Actors import Human
from Environments import Intersection
from Utils import EpisodeGenerator

if __name__ == '__main__':
    np_cars = np.load('Data/Elaborated/np_cars.npy')
    crossing_ped_df = pd.read_pickle('Data/Elaborated/crossing_ped_df.pkl')
    traj_df = pd.read_pickle('Data/Elaborated/traj_df.pkl')

    env = Intersection(np_cars, crossing_ped_df)

    print('Generating episodes')
    ep_gen = EpisodeGenerator(env, traj_df)
    ep_dict, buffer = ep_gen.generate(crossing_ped_df.id, generate_buffer=True)
    buffer.save('Data/Elaborated/exp_buffer')
    print('Episodes generated')
