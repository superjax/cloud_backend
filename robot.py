import numpy as np
from math import *
import matplotlib.pyplot as plt

class Robot():
    def __init__(self, x0, y0, psi0, G):
        self.x = x0
        self.x_true = x0
        self.y = y0
        self.y_true = y0
        self.psi = psi0
        self.psi_true = psi0
        self.G = G

        self.xi = 0
        self.yi = 0
        self.psii = 0

        self.xi_true = 0
        self.yi_true = 0
        self.psii_true = 0

        self.edges = [[x0, y0, psi0]]
        self.true_edges = [[x0, y0, psi0]]
        self.states = [[x0, y0, psi0]]
        self.states_true = [[x0, y0, psi0]]

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



        # Propagate inertial states
        # Calculate Dynamics
        xidot = v * cos(self.psii_true)
        yidot = -v * sin(self.psii_true)
        psidot = w
        self.xi_true += xidot * dt
        self.yi_true += yidot * dt
        self.psii_true += psidot * dt
        # wrap psi to +/- PI
        if self.psii_true > pi:
            self.psii_true -= 2*pi
        elif self.psii_true <= -pi:
            self.psii_true += 2*pi


        xidot = v * cos(self.psii)
        yidot = -v * sin(self.psii)
        psidot = w
        self.xi += (xidot + noise[0]) * dt
        self.yi += (yidot + noise[1]) * dt
        self.psii += (psidot + noise[2]) * dt
        # wrap psi to +/- PI
        if self.psii > pi:
            self.psii -= 2*pi
        elif self.psii <= -pi:
            self.psii += 2*pi


        self.states.append([self.xi, self.yi, self.psii])
        self.states_true.append([self.xi_true, self.yi_true, self.psii_true])

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

        return [x0, y0, 0]

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
        y_true = true_global_state_array[:, 1]

        x_global = np.array(self.states)
        xi = x_global[:,0]
        yi = x_global[:,1]

        x_true_global = np.array(self.states_true)
        xi_true = x_true_global[:,0]
        yi_true = x_true_global[:,1]

        plt.figure(1)
        plt.plot(x, y, label="edges")
        plt.plot(xi, yi, label="states")
        plt.plot(x_true, y_true, label="true edges")
        plt.plot(xi_true, yi_true, label="true states")
        plt.legend()
        plt.show()



















