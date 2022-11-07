import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import LineString, MultiPoint
import seaborn as sns
from typing import Dict


class EpisodeAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def trajs_min_dist(traj1: pd.DataFrame, traj2: pd.DataFrame):
        ts = [t for t in traj1.t.values if t in traj2.t.values]

        traj1.set_index('t', inplace=True)
        traj2.set_index('t', inplace=True)

        dists = [math.sqrt((traj1.loc[t, 'x'] - traj2.loc[t, 'x']) ** 2 + (traj1.loc[t, 'y'] - traj2.loc[t, 'y']) ** 2)
                 for t in ts]

        traj1.reset_index(inplace=True)
        traj2.reset_index(inplace=True)

        if len(dists) == 0:
            return np.inf
        return min(dists)

    def ep_min_dists(self, ep_dict: Dict[int, pd.DataFrame], graph: bool = False):
        min_dist = dict()
        for agent_id in ep_dict.keys():
            ep = ep_dict[agent_id]
            agent_traj = ep.loc[ep.id == agent_id]
            cars_ids = ep.loc[ep.cat == 2, 'id'].unique()
            cars_trajs = [ep.loc[ep.id == car_id] for car_id in cars_ids]
            all_dists = [np.inf]
            for car_traj in cars_trajs:
                all_dists.append(self.trajs_min_dist(agent_traj, car_traj))
            min_dist[agent_id] = min(all_dists)
        min_dist = pd.DataFrame({'ID': min_dist.keys(), 'Minimum car distance': min_dist.values()}).set_index('ID')
        if graph:
            return min_dist, self.generate_graph(min_dist, binwidth=1, xlim=(0, 20), ylim=(0, .25),
                                                 title='Min. distance')
        return min_dist

    def crossing_times(self, ep_dict: Dict[int, pd.DataFrame], graph: bool = False):
        cr_times = dict()
        for agent_id in ep_dict.keys():
            ep = ep_dict[agent_id]
            agent_traj = ep.loc[ep.id == agent_id]
            agent_crossing_time = len(agent_traj.groupby('t').filter(lambda x: abs(x.y) < 5)) / 4
            cr_times[agent_id] = agent_crossing_time
        cr_times = pd.DataFrame({'ID': cr_times.keys(), 'Crossing_times': cr_times.values()}).set_index('ID')
        if graph:
            return cr_times, self.generate_graph(cr_times, binwidth=0.5, xlim=(3, 20), ylim=(0, 0.5),
                                                 title='Crossing time')
        return cr_times

    @staticmethod
    def intersection_time(traj: pd.DataFrame, xi, yi):
        dist = (traj.x.values - xi) ** 2 + (traj.y.values - yi) ** 2
        return traj.iloc[dist.argmin()].t

    def traj_PET(self, traj1: pd.DataFrame, traj2: pd.DataFrame):
        try:
            t1 = LineString(traj1[['x', 'y']].values)
            t2 = LineString(traj2[['x', 'y']].values)
        except:
            return np.NaN

        if not (t1.intersects(t2)):
            return np.NaN

        intersection = t1.intersection(t2)

        if type(intersection) == MultiPoint:
            intersection = intersection.geoms[0]

        # find the time of intersection
        xi = intersection.x
        yi = intersection.y

        t1 = self.intersection_time(traj1, xi, yi)
        t2 = self.intersection_time(traj2, xi, yi)
        return (t1 - t2) / 4

    def ep_min_PET(self, ep_dict: Dict[int, pd.DataFrame], graph: bool = False):
        min_PETs = dict()
        for agent_id in ep_dict.keys():
            ep = ep_dict[agent_id]
            agent_traj = ep.loc[ep.id == agent_id]
            cars_ids = ep.loc[ep.cat == 2, 'id'].unique()
            cars_trajs = [ep.loc[ep.id == car_id] for car_id in cars_ids]
            all_PETs = [np.inf]
            for car_traj in cars_trajs:
                all_PETs.append(self.traj_PET(agent_traj, car_traj))

            min_PETs[agent_id] = min(all_PETs)
        min_PETs = pd.DataFrame({'ID': min_PETs.keys(), 'Minimum PET': min_PETs.values()}).set_index('ID')
        if graph:
            return min_PETs, self.generate_graph(min_PETs, binwidth=2, xlim=(-20, 20), ylim=(0, 0.25), title='PET')
        return min_PETs

    def ep_velocities(self, ep_dict: Dict[int, pd.DataFrame], graph: bool = False):
        velocities = dict()
        for agent_id in ep_dict.keys():
            ep = ep_dict[agent_id]
            agent_traj = ep.loc[ep.id == agent_id]
            velocities[agent_id] = agent_traj[['vx', 'vy']].apply(np.linalg.norm, axis=1).mean()
        velocities = pd.DataFrame({'ID': velocities.keys(), 'Average velocity': velocities.values()}).set_index('ID')
        if graph:
            return velocities, self.generate_graph(velocities, binwidth=0.3, xlim=(0, 4), ylim=(0, 1),
                                                   title='Average velocity')
        return velocities

    @staticmethod
    def velocities_graph(ep_dict: Dict[int, pd.DataFrame]):
        velocities = np.array([])
        for agent_id in ep_dict.keys():
            ep = ep_dict[agent_id]
            agent_traj = ep.loc[ep.id == agent_id]
            velocities = np.concatenate([velocities, agent_traj[['vx', 'vy']].apply(np.linalg.norm, axis=1).values])
        fig, ax = plt.subplots()
        sns.histplot(velocities, kde=True, ax=ax, binwidth=0.3, stat='probability').set(title='Velocities')
        ax.set(ylim=(0, 1), xlim=(0, 4))
        return fig

    @staticmethod
    def traj_graph(ep_dict: Dict[int, pd.DataFrame]):
        img = plt.imread('Data/Elaborated/Intersection.jpg')
        fig, ax = plt.subplots(figsize=(28, 20))
        ax.imshow(img, extent=[-28, 27, -17, 19])
        # Plot cars
        for ep in ep_dict.values():
            car_trajs = ep.loc[ep.cat == 2]
            # Plot car trajs
            for car_id in car_trajs.id.unique():
                car_traj = car_trajs.loc[car_trajs.id == car_id]
                sns.lineplot(x=car_traj.x, y=car_traj.y, sort=False, ax=ax, size=5, legend=None, color='red', alpha=0.1)
        # Plot the agent
        for agent_id in ep_dict.keys():
            ep = ep_dict[agent_id]
            agent_traj = ep.loc[ep.id == agent_id]
            sns.lineplot(x=agent_traj.x, y=agent_traj.y, sort=False, ax=ax, size=5, legend=None, color='green', alpha=0.2)

        ax.set(xlim=(-27, 27), ylim=(-15, 15))
        return fig

    @staticmethod
    def generate_graph(data, binwidth=None, xlim=None, ylim=None, title=None):
        fig, ax = plt.subplots()
        sns.histplot(data, kde=True, ax=ax, binwidth=binwidth, stat='probability').set(title=title)
        ax.set(ylim=ylim, xlim=xlim)
        return fig
