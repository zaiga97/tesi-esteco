import cv2
import numpy as np
import pandas as pd
import torch
from typing import List

from Environments import Intersection
from .Buffers import Buffer


class EpisodeGenerator:
    def __init__(self, env: Intersection, orig_df: pd.DataFrame):
        self.next_states = None
        self.dones = None
        self.rewards = None
        self.actions = None
        self.states = None
        self.env = env
        self.orig_df = orig_df

    def initialize_buffer_lists(self):
        self.states, self.actions, self.rewards, self.dones, self.next_states = [], [], [], [], []

    def add_to_buffer_lists(self, state, action, reward, done, next_state):
        self.states.append(state)
        #if abs(action).max() > 1:
            #action = action / abs(action).max() + 0.05
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def generate_buffer(self):
        self.states = np.array(self.states)
        buffer = Buffer(len(self.states), self.states[0].shape, self.actions[0].shape, 'cpu')
        buffer._n = len(self.states)
        buffer.states = torch.Tensor(self.states).type(torch.float)
        buffer.actions = torch.Tensor(self.actions).type(torch.float)
        buffer.dones = torch.Tensor(self.dones).type(torch.float)
        buffer.next_states = torch.Tensor(self.next_states).type(torch.float)
        buffer.rewards = torch.Tensor(self.rewards).type(torch.float)

        return buffer

    def generate(self, agent, episode_ids: List[int], max_steps: int = 200, generate_films: bool = False,
                 film_path: str = None, generate_buffer: bool = False):

        episodes = dict()
        if generate_buffer:
            self.initialize_buffer_lists()

        for agent_id in episode_ids:
            images = []
            agent_t = []
            agent_x = []
            agent_y = []
            agent_vx = []
            agent_vy = []
            other_agents = set()
            obs, reward, done, ep_len = self.env.reset(agent_id=agent_id), 0, False, 0
            agent.reset(agent_id)

            while (not done) and (ep_len < max_steps):
                t = self.env.t
                other_ids = self.orig_df.loc[self.orig_df.t == t, 'id'].values
                for other_id in other_ids:
                    other_agents.add(other_id)

                action = agent.exploit(obs)
                agent_vx.append(action[0] * self.env.step_scale * 4)
                agent_vy.append(action[1] * self.env.step_scale * 4)

                new_obs, reward, done, _ = self.env.step(action)

                if generate_buffer:
                    self.add_to_buffer_lists(obs, action, reward, done, new_obs)

                if generate_films:
                    images.append(self.env.render().astype(np.uint8))

                obs = new_obs
                agent_t.append(t)
                agent_x.append(obs[0])
                agent_y.append(obs[1])
                ep_len += 1

            other_agents.remove(agent_id)
            other_agents_df = self.orig_df.loc[self.orig_df.id.isin(other_agents)].copy()
            if self.env.flip:
                other_agents_df[['x', 'vx', 'ax', 'y', 'vy', 'ay']] *= -1
            episodes[agent_id] = pd.concat([other_agents_df,
                                            pd.DataFrame({'id': agent_id, 't': agent_t, 'x': agent_x, 'y': agent_y,
                                                          'vx': agent_vx, 'vy': agent_vy})])
            if generate_films:
                y_size, x_size, _ = images[0].shape
                writer = cv2.VideoWriter(f"{film_path}/{agent_id}.avi", cv2.VideoWriter_fourcc(*"MJPG"), 8,
                                         (x_size, y_size))
                for img in images:
                    writer.write(img)
                writer.release()
        if generate_buffer:
            return episodes, self.generate_buffer()
        return episodes
