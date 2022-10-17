import copy

import pandas as pd
import torch

from Actors import Actor
from Environments import Intersection
import numpy as np
import cv2 as cv


class FilmMaker:
    def __init__(self):
        self.x_pixels = 700
        self.y_pixels = 500

        def to_render_coord(coordinates: np.array):
            x, y = coordinates[0], coordinates[1]
            x = int((x + 35) / 70 * self.x_pixels)
            y = int((y + 25) / 50 * self.y_pixels)
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

    def film(self, ep_dict: dict[int, pd.DataFrame], out_path: str = ''):
        for agent_id in ep_dict.keys():
            ep = ep_dict[agent_id]
            images = []
            for t in ep.t.unique().sort():
                img = self.draw_img(ep.loc[ep.t == t], agent_id)
                images.append(img)

            y_size, x_size, _ = images[0].shape
            name = f"{out_path}/ep_{agent_id}"
            writer = cv.VideoWriter(name + ".avi", cv.VideoWriter_fourcc(*"MJPG"), 10, (x_size, y_size))
            for img in images:
                writer.write(img)
            writer.release()

    def draw_img(self, state: pd.DataFrame, agent_id: int):
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
