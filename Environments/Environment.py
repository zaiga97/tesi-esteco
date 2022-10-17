import copy
from typing import Tuple

import cv2 as cv
import numpy as np
import pandas as pd
from gym import spaces, Env


class Intersection(Env):
    def __init__(self, np_cars: np.array, pedestrian_df: pd.DataFrame, max_step: float = .6):
        super(Intersection, self).__init__()
        self.max_step = max_step
        self.np_cars = np_cars
        self.pedestrian_df = pedestrian_df.set_index('id')
        self.agent_id = None
        self.t = None
        self.agent_pos = np.array([None, None], dtype=np.float32)
        self.target_pos = np.array([None, None], dtype=np.float32)
        self.state = np.array([None])
        self.reward = 0
        self.progress = 0
        self.last_action = np.array([0, 0], dtype=np.float32)
        self.closer_car_vec = np.array([np.inf, np.inf])
        self.done = True
        # Action space = {dx, dy}
        self.action_space = spaces.Box(np.array([-self.max_step, -self.max_step]).astype(np.float32),
                                       np.array([+self.max_step, +self.max_step]).astype(np.float32))
        # Observation space = {x, y, xt, yt, x1, y1, vx1, ax1, x2, y2, vx2, ax2}
        # The 1 is the closest car to the lower intersection that has still to pass it
        # The 2 is the closest car to the upper intersection that has still to pass it
        xm, ym, xM, yM, vm, vM, am, aM = -35, -25, 35, 25, -50, 50, -20, 20
        self.observation_space = spaces.Box(
            low=np.array([xm, ym, xm, ym, xm, ym, vm, am, xm, ym, vm, am]).astype(np.float32),
            high=np.array([xM, yM, xM, yM, xM, yM, vM, aM, xM, yM, vM, aM]).astype(np.float32))
        self.x_pixels = 700
        self.y_pixels = 500

        def to_render_coord(coordinates: np.array):
            x, y = coordinates[0], coordinates[1]
            x = int((x - xm) / (xM - xm) * self.x_pixels)
            y = int((y - ym) / (yM - ym) * self.y_pixels)
            return x, y

        self.to_render_coord = to_render_coord
        self.renderSpace = (self.y_pixels, self.x_pixels, 3)
        self.canvas = np.ones(self.renderSpace) * 255
        self.init_canvas()

    def init_canvas(self):
        # Street
        cv.rectangle(self.canvas, self.to_render_coord(np.array([-35, -5])), self.to_render_coord(np.array([35, 5])),
                     (0, 0, 0), cv.FILLED)
        cv.rectangle(self.canvas, self.to_render_coord(np.array([-35, -.05])),
                     self.to_render_coord(np.array([35, .05])),
                     (124, 124, 124), cv.FILLED)
        # Crossing
        cv.rectangle(self.canvas, self.to_render_coord(np.array([-2, -5])), self.to_render_coord(np.array([2, 5])),
                     (124, 124, 124), cv.FILLED)
        # Safe area
        cv.rectangle(self.canvas, self.to_render_coord(np.array([-2, -.5])), self.to_render_coord(np.array([2, .5])),
                     (255, 255, 255), cv.FILLED)

    def update_state(self) -> None:
        self.state = np.concatenate([self.agent_pos, self.target_pos, self.np_cars[self.t][0], self.np_cars[self.t][1]])
        self.closer_car_vec = min([self.agent_pos - self.state[4: 6], self.agent_pos - self.state[8: 10]],
                                  key=lambda x: np.linalg.norm(x))
        self.done = self.check_if_done()
        self.reward = self.calculate_reward()

    def step(self, action: np.array) -> Tuple[np.array, float, bool, dict]:
        # assert self.action_space.contains(action), "Action is not in the action space"
        self.t += 1
        self.last_action = action
        old_dist = np.linalg.norm(self.agent_pos - self.target_pos)
        self.agent_pos += action
        new_dist = np.linalg.norm(self.agent_pos - self.target_pos)
        self.progress = (old_dist - new_dist) / self.max_step
        self.update_state()
        return self.state, self.reward, self.done, {}

    def reset(self, agent_id: int = None, **kwargs):
        if agent_id is None:
            agent_id = np.random.choice(self.pedestrian_df.index)

        self.agent_id = agent_id
        self.t = self.pedestrian_df.loc[agent_id, 'ts']
        self.agent_pos = self.pedestrian_df.loc[agent_id, ['xs', 'ys']].values
        self.target_pos = self.pedestrian_df.loc[agent_id, ['xf', 'yf']].values

        self.update_state()
        return self.state

    def render(self, **kwargs):
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLUE = (255, 0, 0)

        canvas = copy.deepcopy(self.canvas)

        # Draw the agent
        cv.circle(canvas, (self.to_render_coord(self.agent_pos)), 4, GREEN, cv.FILLED)
        cv.circle(canvas, (self.to_render_coord(self.target_pos)), 6, BLUE, cv.FILLED)
        # Draw car 1
        p1 = np.array([self.state[4] - 2, self.state[5] + 1])
        p2 = np.array([self.state[4] + 2, self.state[5] - 1])
        cv.rectangle(canvas, self.to_render_coord(p1), self.to_render_coord(p2), RED, cv.FILLED)
        # Draw car 2
        p1 = np.array([self.state[8] - 2, self.state[9] + 1])
        p2 = np.array([self.state[8] + 2, self.state[9] - 1])
        cv.rectangle(canvas, self.to_render_coord(p1), self.to_render_coord(p2), RED, cv.FILLED)
        # Add reward
        cv.putText(canvas, str(self.reward), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

        return canvas

    def calculate_reward(self):
        reward = 0
        # Time penalty
        # reward -= 0.025
        # Progress term
        reward += self.progress
        if self.progress < -0.1:
            reward -= 1
        # Speed term
        reward -= ((np.linalg.norm(self.last_action) / self.max_step) ** 4) * 1.5
        # To close to cars penalty
        if abs(self.agent_pos[1]) > 0.7:
            if abs(self.closer_car_vec[0]) < 2.8 and abs(self.closer_car_vec[1]) < 1.3:
                reward -= 20
        # Outside designed wander space
        if abs(self.agent_pos[1]) < 5 and abs(self.agent_pos[0]) > 1.6:
            reward -= 5
        # If you reach the end get a reward
        if self.done:
            reward += 80
        return reward

    def check_if_done(self):
        return np.linalg.norm(self.agent_pos - self.target_pos) < 1
