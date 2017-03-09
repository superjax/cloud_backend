import numpy as np
from enum import Enum

class Controller_State(Enum):
    TURNING = 0
    STRAIGHT = 1

class Controller:
    def __init__(self):
        self.state = Controller_State.STRAIGHT
        self.switch_probability = 0.01
        self.vel_noise = 0.2
        self.omega_noise = 0.2
        self.turn_direction = 1
        self.nominal_omega = 1.0
        self.nominal_velocity = 1.0

    def control(self, t):
        if np.random.uniform(0, 1.0) < self.switch_probability:
            if self.state == Controller_State.STRAIGHT:
                self.state = Controller_State.TURNING
                self.turn_direction = np.sign(np.random.uniform(-0.5, 0.5))
            else:
                self.state = Controller_State.STRAIGHT



        if self.state == Controller_State.STRAIGHT:
            return [self.nominal_velocity + np.random.normal(0, self.vel_noise),
                    0.0 + np.random.normal(0, self.omega_noise)]
        if self.state == Controller_State.TURNING:
            return [self.nominal_velocity + np.random.normal(0, self.vel_noise),
                    self.nominal_omega * self.turn_direction + np.random.normal(0, self.omega_noise)]
