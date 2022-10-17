import numpy as np
import pandas as pd

from Actors import ActorCritic
from Environments import Intersection
from Algorithms import ddpg

if __name__ == '__main__':
    np_cars = np.load('Data/Elaborated/np_cars.npy')
    crossing_ped_df = pd.read_pickle('Data/Elaborated/crossing_ped_df.pkl')
    traj_df = pd.read_pickle('Data/Elaborated/traj_df.pkl')

    PATH = "Data/Trained_Models/RL_agent_00"

    env = Intersection(np_cars, crossing_ped_df)
    ac = ActorCritic(env.observation_space, env.action_space)


    def env_fn():
        return env

    print("Training the agent: ...")
    ddpg(env_fn, actor_critic=ac, epochs=100, max_ep_len=200)
    print("Agent trained: ...")
    print(f"Saving model to: {PATH}")
    ac.save(PATH)
    print(f"Model saved")
