import cv2
import numpy as np
import pandas as pd
import torch
from typing import List

from Actors import Actor
from Environments import Intersection
from Utils import FilmMaker


class EpisodeGenerator:
    def __init__(self, env: Intersection, orig_df: pd.DataFrame):
        self.env = env
        self.orig_df = orig_df
        self.film_maker = FilmMaker()

    @staticmethod
    def generate_from_df(episode_ids: List[int], orig_df: pd.DataFrame):
        episodes = dict()
        for agent_id in episode_ids:
            ts = orig_df.loc[orig_df.id == agent_id, 't']
            episodes[agent_id] = orig_df.loc[orig_df.t.isin(ts)]
        return episodes

    def generate(self, episode_ids: List[int], agent: Actor = None, max_steps: int = 200, generate_films: bool = False,
                 film_path: str = None):
        if agent is None:
            return self.generate_from_df(episode_ids, self.orig_df)
        episodes = dict()

        for agent_id in episode_ids:
            images = []
            agent_t = []
            agent_x = []
            agent_y = []
            agent_vx = []
            agent_vy = []
            other_agents = set()
            obs, done, ep_len = self.env.reset(agent_id=agent_id), False, 0
            agent.reset()

            while (not done) and (ep_len < max_steps):
                t = self.env.t
                other_ids = self.orig_df.loc[self.orig_df.t == t, 'id'].values
                for other_id in other_ids:
                    other_agents.add(other_id)

                action = agent.act(torch.Tensor(obs))
                agent_vx.append(action[0]*4)
                agent_vy.append(action[1]*4)
                obs, reward, done, _ = self.env.step(action)
                if generate_films:
                    images.append(self.env.render().astype(np.uint8))

                agent_t.append(t)
                agent_x.append(obs[0])
                agent_y.append(obs[1])
                ep_len += 1

            other_agents.remove(agent_id)
            episodes[agent_id] = pd.concat([self.orig_df.loc[self.orig_df.id.isin(other_agents)],
                                            pd.DataFrame({'id': agent_id, 't': agent_t, 'x': agent_x, 'y': agent_y,
                                                          'vx': agent_vx, 'vy': agent_vy})])
            if generate_films:
                y_size, x_size, _ = images[0].shape
                writer = cv2.VideoWriter(f"{film_path}/{agent_id}.avi", cv2.VideoWriter_fourcc(*"MJPG"), 8,
                                         (x_size, y_size))
                for img in images:
                    writer.write(img)
                writer.release()
        return episodes
