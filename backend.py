#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from math import *
import numpy as np
import itertools
import regex
class Node():
    def __init__(self, id, pose):
        self.id = id
        self.pose = pose
        self.true_pose = pose

    def set_pose(self, pose):
        self.pose = pose

class Edge():
    def __init__(self, from_id, to_id, covariance, transform):
        self.from_id = from_id
        self.to_id = to_id
        self.covariance = covariance
        self.transform = transform

class Backend():
    def __init__(self, name="default"):
        self.name = name
        self.G = nx.Graph()
        self.node_plot_positions = dict()
        self.lc_edges = []
        self.LC_threshold = 2.5

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
            x0 = from_pose[0]
            y0 = from_pose[1]
            psi0 = from_pose[2]
            dx = edge.transform[0]
            dy = edge.transform[1]
            dpsi = edge.transform[2]
            x1 = x0 + dx * cos(psi0) - dy * sin(psi0)
            y1 = y0 + dx * sin(psi0) + dy * cos(psi0)
            psi1 = psi0 + dpsi
            # wrap psi to +/- PI
            if psi1 > pi:
                psi1 -= 2 * pi
            elif psi1 <= -pi:
                psi1 += 2 * pi

            self.G.node[edge.to_id]['pose'] = [x1, y1, psi1]
            self.node_plot_positions[edge.to_id] = [y1, x1]
        if not edge.to_id == edge.from_id +1:
            from_pose = self.G.node[edge.from_id]['pose']
            to_pose = self.G.node[edge.to_id]['pose']
            self.lc_edges.append((from_pose, to_pose))

    def simulate_loop_closures(self):
        loop_closures = []
        # compare all nodes, and see if the poses are similar
        for (i, n), (j,m) in itertools.combinations(self.G.node.iteritems(), 2):
            # don't find loop closures between close nodes
            if abs(i - j) < 10:
                continue
            if j < i:
                continue
            n_pose = np.array(n['pose'])
            m_pose = np.array(m['pose'])
            if np.linalg.norm(n_pose - m_pose) < self.LC_threshold:
                P = [[0.1, 0, 0],
                     [0, 0.1, 0],
                     [0, 0, 0.1]]
                transform = (m_pose - n_pose).tolist()
                lc = Edge(i, j, P, transform)
                self.add_edge(lc)
                loop_closures.append(lc)
                self.lc_edges.append((n['pose'], m['pose']))
        return loop_closures

    def output_g2o(self, filename):
        f = open(filename, 'w')
        for i in self.G.nodes_iter():
            line = "VERTEX_SE2 " + str(i) + " " + \
                    str(self.G.node[i]['pose'][0]) + " " + \
                    str(self.G.node[i]['pose'][1]) + " " + \
                    str(self.G.node[i]['pose'][2]) + "\n"
            f.write(line)
        f.write("FIX 0\n")
        for i, j in self.G.edges_iter():
            line = "EDGE_SE2 " + str(i) + " " + str(j) + \
                    " " + str(self.G.edge[i][j]['transform'][0]) + \
                    " " + str(self.G.edge[i][j]['transform'][1]) + \
                    " " + str(self.G.edge[i][j]['transform'][2]) + \
                    " " + str(self.G.edge[i][j]['covariance'][0][0]) + \
                    " " + str(self.G.edge[i][j]['covariance'][0][1]) + \
                    " " + str(self.G.edge[i][j]['covariance'][0][2]) + \
                    " " + str(self.G.edge[i][j]['covariance'][1][1]) + \
                    " " + str(self.G.edge[i][j]['covariance'][1][2]) + \
                    " " + str(self.G.edge[i][j]['covariance'][2][2]) + "\n"
            f.write(line)

    def load_g2o(self, edges_file):
        f_e = open(edges_file, 'r')
        for line in f_e:
            match = regex.search("EDGE_SE2", line)
            if match != None:
                e_str = line.split(" ")
                from_id = int(e_str[1])
                to_id = int(e_str[2])
                x = float(e_str[3])
                y = float(e_str[4])
                theta = float(e_str[5])
                covariance = [[float(e_str[6]), float(e_str[7]), float(e_str[8])],
                              [float(e_str[7]), float(e_str[9]), float(e_str[10])],
                              [float(e_str[8]), float(e_str[10]), float(e_str[11])]]
                e = Edge(from_id, to_id, covariance, [x, y, theta])
                self.add_edge(e)

    def plot_graph(self, arrows=True, figure_handle=0, edge_color='k', lc_color='y'):
        if figure_handle:
            plt.figure(figure_handle)
        else:
            plt.figure()
        self.ax = plt.subplot(111)
        self.ax.axis('equal')
        plt.title(self.name)
        nx.draw_networkx(self.G, pos=self.node_plot_positions,
                         with_labels=False, ax=self.ax, edge_color=edge_color,
                         linewidths="0.3", node_color='c', node_shape='x')
        if arrows:
            for i, n in self.G.node.iteritems():
                pose = n['pose']
                arrow_length = 0.05
                dx = arrow_length * cos(pose[2])
                dy = arrow_length * sin(pose[2])
                # Be sure to convert to NWU for plotting
                self.ax.arrow(pose[1], pose[0], dy, dx,
                         head_width=0.015, head_length=0.03, fc='c', ec='b')
                plt.axis('equal')

        # Plot loop closures
        for lc in self.lc_edges:
            x = [lc[0][0], lc[1][0]]
            y = [lc[0][1], lc[1][1]]
            self.ax.plot(y, x , lc_color, lw='1.5', zorder=1)

