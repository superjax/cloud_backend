import numpy as np
from math import *
import matplotlib.pyplot as plt

class Robot():
    def __init__(self, id, G):
        self.id = id
        self.x = 0
        self.x_true = 0
        self.y = 0
        self.y_true = 0
        self.psi = 0
        self.psi_true = 0
        self.G = G

        self.edges = []
        self.true_edges = []

    def propagate_dynamics(self, u, dt):
        noise = np.random.multivariate_normal(np.array([0, 0, 0]), self.G)
        v = u[0]
        w = u[1]

        # Calculate Dynamics
        xdot = v * cos(self.psi)
        ydot = -v * sin(self.psi)
        psidot = w

        # Euler Integration (noisy)
        self.x +=   (xdot + noise[0]) * dt
        self.y +=   (ydot + noise[1]) * dt
        self.psi += (psidot + noise[2]) * dt
        # wrap psi to +/- PI
        if self.psi > pi:
            self.psi -= 2*pi
        elif self.psi <= -pi:
            self.psi += 2*pi

        # Propagate truth
        self.x_true += xdot * dt
        self.y_true += ydot * dt
        self.psi_true += psidot * dt
        # wrap psi to +/- PI
        if self.psi_true > pi:
            self.psi_true -= 2*pi
        elif self.psi_true <= -pi:
            self.psi_true += 2*pi

        return np.array([[self.x, self.y, self.psi]]).T

    def reset(self):
        edge = [self.x, self.y, self.psi]
        true_edge = [self.x_true, self.y_true, self.psi_true]
        self.edges.append(edge)
        self.true_edges.append(true_edge)

        # reset state
        self.x = 0
        self.y = 0
        self.psi = 0

        self.x_true = 0
        self.y_true = 0
        self.psi_true = 0

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

    def find_true_global_state(self):
        combined_edge = [self.true_edges[0]]
        for edge in self.true_edges[1:]:
            combined_edge.append(self.concatenate_edges(combined_edge[-1], edge))
        self.true_global_state = combined_edge
        return np.array(combined_edge)

    def draw_trajectory(self):
        global_state_array = self.find_global_state()
        true_global_state_array = self.find_true_global_state()
        x = global_state_array[:,0]
        y = global_state_array[:,1]

        x_true = true_global_state_array[:,0]
        y_true = true_global_state_array[:,1]

        plt.figure()
        plt.plot(y, x, label="edges")
        plt.plot(y_true, x_true, label="true edges")
        plt.axis("equal")
        plt.legend()
        plt.show()



















