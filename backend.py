#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from math import *

class Node():
    def __init__(self, id, pose):
        self.id = id
        self.pose = pose

    def set_pose(self, pose):
        self.pose = pose

class Edge():
    def __init__(self, from_id, to_id, covariance, transform):
        self.from_id = from_id
        self.to_id = to_id
        self.covariance = covariance
        self.transform = transform

class Backend():
    def __init__(self):
        self.G = nx.Graph()
        self.node_plot_positions = dict()
        self.ax = plt.subplot(111)

    # def add_node(self, node):
    #     id = node.id
    #     pose = node.pose
    #     self.G.add_node(id, pose=pose)
    #     self.node_plot_positions[id] = pose[0:2]


    def add_edge(self, edge):
        self.G.add_edge(edge.from_id, edge.to_id, covariance=edge.covariance, transform=edge.transform)
        if edge.from_id == 0:
            self.G.node[edge.from_id]['pose'] = [0, 0, 0]
            self.node_plot_positions[0] = [0, 0]
        elif not 'pose' in self.G.node[edge.from_id]:
            raise NameError('Undetermined edge origin')
        if not 'pose' in self.G.node[edge.to_id]:
            from_pose = self.G.node[edge.from_id]['pose']

            # concatenate previous node position with edge transform
            x0 = from_pose[0]
            y0 = from_pose[1]
            psi0 = from_pose[2]
            dx = edge.transform[0]
            dy = edge.transform[1]
            dpsi = edge.transform[2]
            x1 = x0 + dx * cos(psi0) - dy * sin(psi0)
            y1 = y0 + dx * sin(psi0) + dy * cos(psi0)
            psi1 = psi0 + dpsi

            self.G.node[edge.to_id]['pose'] = [x1, y1, psi1]
            self.node_plot_positions[edge.to_id] = [y1, x1]






    def plot_graph(self):
        nx.draw_networkx(self.G, pos=self.node_plot_positions,
                         with_labels=True, ax=self.ax, edge_color='k',
                         linewidths="0.3", node_color='c')
        for i, n in self.G.node.iteritems():
            pose = n['pose']
            arrow_length = 0.2
            dx = arrow_length * cos(pose[2])
            dy = arrow_length * sin(pose[2])
            # Be sure to convert to NWU for plotting
            self.ax.arrow(pose[1], pose[0], dy, dx,
                     head_width=0.03, head_length=0.06, fc='c', ec='b')
        plt.show()

