#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from math import *
import numpy as np
import itertools
import regex
from tqdm import tqdm

class Node():
    def __init__(self, id, pose):
        self.id = id
        self.pose = pose
        self.true_pose = pose

    def set_pose(self, pose):
        self.pose = pose

class Edge():
    def __init__(self, vehicle_id, from_id, to_id, covariance, transform):
        self.from_id = from_id
        self.to_id = to_id
        self.covariance = covariance
        self.transform = transform
        self.vehicle_id = vehicle_id

class Backend():
    def __init__(self, name="default"):
        self.name = name
        self.G = nx.DiGraph()
        self.node_plot_positions = dict()
        self.lc_edges = []
        self.LC_threshold = 1.5
        self.overlap_threshold = 0.75
        self.node_id_map = dict()

    def add_agent(self, vehicle_id, start_pose):
        # Create a new starting node for this agent
        node_id = str(vehicle_id) + "_" + str(0)
        self.G.add_node(node_id, pose=start_pose, vehicle_id=vehicle_id)
        self.node_plot_positions[node_id] = [start_pose[1], start_pose[0]]

    def add_edge(self, edge):
        # save off useful variables
        vehicle_id = edge.vehicle_id
        from_id = edge.from_id
        to_id = edge.to_id

        # Add this edge to the networkx graph
        self.G.add_edge(from_id, to_id, covariance=edge.covariance,
                        transform=edge.transform)

        # We can't calculate the pose of the nodes if we don't have a pose
        # of the other side
        if not 'pose' in self.G.node[from_id]:
            raise NameError('Undetermined edge origin')

        # Calculate a best guess of where the new nodes position
        if not 'pose' in self.G.node[to_id]:
            from_pose = self.G.node[from_id]['pose']
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

            # Populate the attributes of the new node
            self.G.node[to_id]['pose'] = [x1, y1, psi1]
            self.G.node[to_id]['vehicle_id'] = vehicle_id
            self.node_plot_positions[to_id] = [y1, x1]

        # If these are not consecutive, then this is a loop closure
        # That means we don't need to calculate plotting position for
        # the new node, and we should add it to the lc plotting array
        if not abs(int(edge.to_id.split("_")[1]) - int(edge.from_id.split("_")[1])) == 1:
            from_pose = self.G.node[from_id]['pose']
            to_pose = self.G.node[to_id]['pose']
            self.lc_edges.append((from_pose, to_pose))

    def simulate_loop_closures(self):
        loop_closures = []
        # compare all nodes, and see if the poses are similar
        for (i, m), (j, n) in tqdm(itertools.combinations(self.G.node.iteritems(), 2)):
            # don't find loop closure between odometry
            if n['vehicle_id'] == m['vehicle_id']:
                # Make sure that these edges are far apart
                if abs(int(i.split("_")[1]) - int(j.split("_")[1])) < 10:
                    continue

            m_pose = np.array(m['pose'])
            n_pose = np.array(n['pose'])
            if np.linalg.norm(n_pose - m_pose) < self.LC_threshold:
                if i == '0_140':
                    debug = 1
                P = [[0.0000001, 0, 0],
                     [0, .0000001, 0],
                     [0, 0, .0000001]]
                # Rotate transform frame to the "from_node" frame
                xij_I = n_pose[0:2] - m_pose[0:2]
                psii = m_pose[2]
                R_I_to_i = np.array([[cos(psii), sin(psii)],
                                     [-sin(psii), cos(psii)]])
                dx = xij_I.dot(R_I_to_i.T)
                dpsi = n_pose[2] - psii
                lc = Edge(n['vehicle_id'], i, j, P, [dx[0], dx[1], dpsi])
                self.add_edge(lc)
                loop_closures.append(lc)
                self.lc_edges.append((n['pose'], m['pose']))
        return loop_closures

    def output_g2o(self, filename):
        node_id_map = dict()
        node_id_map['g2o_index'] = dict()
        node_id_map['vID_index'] = dict()
        f = open(filename, 'w')
        g2o_index = 0
        for i in self.G.nodes_iter():
            node_id_map['g2o_index'][g2o_index] = i
            node_id_map['vID_index'][i] = g2o_index
            line = "VERTEX_SE2 " + str(g2o_index) + " " + \
                str(self.G.node[i]['pose'][0]) + " " + \
                str(self.G.node[i]['pose'][1]) + " " + \
                str(self.G.node[i]['pose'][2]) + "\n"

            f.write(line)
            g2o_index += 1
        f.write("FIX " + str(node_id_map['vID_index']['0_0']) + "\n")
        for i, edge in self.G.adjacency_iter():
            for j in edge.iterkeys():
                line = "EDGE_SE2 " + str(node_id_map['vID_index'][i]) + \
                        " " + str(node_id_map['vID_index'][j]) + \
                        " " + str(self.G.edge[i][j]['transform'][0]) + \
                        " " + str(self.G.edge[i][j]['transform'][1]) + \
                        " " + str(self.G.edge[i][j]['transform'][2]) + \
                        " " + str(1.0/self.G.edge[i][j]['covariance'][0][0]) + \
                        " " + str(self.G.edge[i][j]['covariance'][0][1]) + \
                        " " + str(self.G.edge[i][j]['covariance'][0][2]) + \
                        " " + str(1.0/self.G.edge[i][j]['covariance'][1][1]) + \
                        " " + str(self.G.edge[i][j]['covariance'][1][2]) + \
                        " " + str(1.0/self.G.edge[i][j]['covariance'][2][2]) + "\n"
                f.write(line)
        return node_id_map

    def load_g2o(self, g2o_file, node_id_map):
        # G2O Doesn't modify edges, so we have to re-calculate edges from the optimized
        # node positions
        f_e = open(g2o_file, 'r')
        nodes = dict()
        edges = []
        for line in f_e:
            if regex.search("VERTEX_SE2", line) != None:
                n_str = line.split(" ")
                id = node_id_map['g2o_index'][int(n_str[1])]
                x = float(n_str[2])
                y = float(n_str[3])
                theta = float(n_str[4])
                nodes[id] = [x, y, theta]
                self.G.add_node(id, pose=[x, y, theta], vehicle_id=id.split("_")[0])
                self.node_plot_positions[id] = [y, x]
            elif regex.search("EDGE_SE2", line) != None:
                e_str = line.split(" ")
                from_id = node_id_map['g2o_index'][int(e_str[1])]
                to_id = node_id_map['g2o_index'][int(e_str[2])]
                self.G.add_edge(from_id, to_id)
                edges.append([from_id, to_id])
                if abs(int(from_id.split("_")[1]) - int(to_id.split("_")[1])) > 1:
                    from_pose = self.G.node[from_id]['pose']
                    to_pose = self.G.node[to_id]['pose']
                    self.lc_edges.append((from_pose, to_pose))

    def plot_graph(self, arrows=True, figure_handle=0, edge_color='k', lc_color='y'):
        if figure_handle:
            plt.figure(figure_handle)
        else:
            plt.figure()
        self.ax = plt.subplot(111)
        plt.title(self.name)
        nx.draw_networkx(self.G, pos=self.node_plot_positions,
                         with_labels=False, ax=self.ax, edge_color=edge_color,
                         linewidths="0.3", node_color='c', node_shape='')
        if arrows:
            for i, n in self.G.node.iteritems():
                pose = n['pose']
                arrow_length = 0.05
                dx = arrow_length * cos(pose[2])
                dy = arrow_length * sin(pose[2])
                # Be sure to convert to NWU for plotting
                self.ax.arrow(pose[1], pose[0], dy, dx,
                         head_width=0.015, head_length=0.03, fc='c', ec='b')

        # Plot loop closures
        for lc in self.lc_edges:
            x = [lc[0][0], lc[1][0]]
            y = [lc[0][1], lc[1][1]]
            self.ax.plot(y, x , lc_color, lw='0.1', zorder=1)

        plt.axis("equal")


