import numpy as np
from math import *
import matplotlib.pyplot as plt

class Robot():
    def __init__(self, x0, y0, psi0, Q):
        self.x = x0
        self.y = y0
        self.psi = psi0
        self.Q = Q

        self.xi = x0
        self.yi = y0
        self.psii = psi0

        self.edges = [[x0, y0, psi0]]
        self.states = [[x0, y0, psi0]]

    def propagate_dynamics(self, u, dt):
        v = u[0]
        w = u[1]

        # Calculate Dynamics
        xdot = v * cos(self.psi)
        ydot = -v * sin(self.psi)
        psidot = w

        # Euler Integration
        self.x += xdot * dt
        self.y += ydot * dt
        self.psi += psidot * dt

        # Propagate inertial states
        # Calculate Dynamics
        xidot = v * cos(self.psii)
        yidot = -v * sin(self.psii)
        psidot = w
        self.xi += xidot * dt
        self.yi += yidot * dt
        self.psii += psidot * dt

        self.states.append([self.xi, self.yi, self.psii])

        return np.array([[self.x, self.y, self.psi]]).T

    def reset(self):
        edge = [self.x, self.y, self.psi]
        self.edges.append(edge)

        # reset state
        self.x = 0
        self.y = 0
        self.psi = 0

    def concatenate_edges(self, edge1, edge2):
        x0 = edge1[0]
        x1 = edge2[0]

        y0 = edge1[1]
        y1 = edge2[1]

        psi0 = edge1[2]
        psi1 = edge2[2]

        #concatenate edges by rotating second edge into the first edge's frame
        x0 += x1*cos(psi0) + y1*sin(psi0)
        y0 += -x1*sin(psi0) + y1*cos(psi0)
        psi0 += psi1

        return [x0, y0, psi0]

    def find_global_state(self):
        combined_edge = [self.edges[0]]
        for edge in self.edges[1:]:
            combined_edge.append(self.concatenate_edges(combined_edge[-1], edge))
        self.global_state = combined_edge
        return np.array(combined_edge)

    def draw_trajectory(self):
        global_state_array = self.find_global_state()
        x = global_state_array[:,0]
        y = global_state_array[:,1]

        x_global = np.array(self.states)
        xi = x_global[:,0]
        yi = x_global[:,1]

        plt.figure(1)
        plt.plot(x, y, label="edges")
        plt.plot(xi, yi, label="states")
        plt.show()



















