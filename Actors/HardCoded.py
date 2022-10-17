import numpy as np
import torch

from Utils import RoadGeometry as rg
from .Actor import Actor


class HardCoded(Actor):
    def __init__(self, max_step: float = .35, straight_to_target: bool = False, check_traffic: bool = True):
        super(HardCoded, self).__init__()
        self.target = None
        self.A = torch.Tensor([0, -rg.LOW_Y_INTERSECTION])
        self.B = torch.Tensor([0, 0])
        self.C = torch.Tensor([0, rg.HIGH_Y_INTERSECTION])
        self.got_A = self.got_B = self.got_C = straight_to_target
        self.check = False
        self.max_step = max_step
        self.straight_to_target = straight_to_target
        self.check_traffic = check_traffic

    def act(self, state: torch.Tensor):
        position = state[0: 2]
        # Check if we reached current target
        if self.target is not None:
            dt = np.linalg.norm(position - self.target)
            if dt < self.max_step:
                if not self.got_A:
                    self.got_A = True
                    self.check = self.check_traffic
                elif not self.got_B:
                    self.got_B = True
                    self.check = self.check_traffic
                elif not self.got_C:
                    self.got_C = True

                self.target = None

        # Update the target if needed
        if self.target is None:
            if not self.got_A:
                dA = np.linalg.norm(position - self.A)
                dC = np.linalg.norm(position - self.C)
                # Swap A and C if needed
                if dA > dC:
                    (self.A, self.C) = (self.C, self.A)
                self.target = self.A

            elif not self.got_B:
                self.target = self.B

            elif not self.got_C:
                self.target = self.C

            else:
                self.target = state[2: 4]

        # Calculate the current vector to target
        target_v = self.target - position

        # Check for traffic
        if self.check:
            crossing_time = (np.linalg.norm(target_v) / self.max_step) / 4
            y = position[1]
            yt = self.target[1]
            # Check the lower
            if (y < 0 and not self.got_B) or (self.got_B and yt < 0):
                xc = state[8]
                vc = state[10]
                ac = state[11]
                if not (xc < -2):
                    if (xc + (vc * crossing_time)) < 2:
                        return np.array([0, 0])

            elif (y > 0 and not self.got_B) or (self.got_B and yt > 0):
                xc = state[4]
                vc = state[6]
                ac = state[7]
                if not (xc > 2):
                    if (xc + (vc * crossing_time)) > -2:
                        return np.array([0, 0])
            # If we reach here is because there are no predicted collision
            self.check = False

        return np.array((target_v / np.linalg.norm(target_v)) * self.max_step)

    def reset(self):
        super(HardCoded, self).reset()
        self.target = None
        self.got_A = self.got_B = self.got_C = self.check = self.straight_to_target
