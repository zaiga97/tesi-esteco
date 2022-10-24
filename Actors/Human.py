import numpy as np
import pandas as pd
import torch


class Human():
    def __init__(self, orig_trajs: pd.DataFrame):
        super().__init__()
        self.orig_trajs = orig_trajs
        self.current_traj = None
        self.t = None

    def act(self, *args):
        self.t += 1
        if self.t in self.current_traj.t.values:
            return self.current_traj.loc[self.current_traj.t == self.t, ['vx', 'vy']].values[0]/4
        else:
            return np.array([0, 0])


    def reset(self, agent_id):
        self.current_traj = self.orig_trajs.loc[self.orig_trajs.id == agent_id]
        self.t = self.current_traj.t.min()
