import numpy as np
import pandas as pd

from Actors import Human
from Environments import make_intersection
from Utils import EpisodeGenerator

if __name__ == '__main__':
    np_cars = np.load('Data/Elaborated/np_cars.npy')
    crossing_ped_df = pd.read_pickle('Data/Elaborated/crossing_ped_df.pkl')
    traj_df = pd.read_pickle('Data/Elaborated/traj_df.pkl')

    env = make_intersection()
    env.step_scale = 1.
    ep_gen = EpisodeGenerator(env, traj_df)
    actor = Human(traj_df, env.step_scale)

    _, buffer = ep_gen.generate(actor, crossing_ped_df.id, generate_buffer=True)
    buffer.save('Data/Elaborated/exp_buffer')
