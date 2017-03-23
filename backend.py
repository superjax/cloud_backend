#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from math import *
import numpy as np
import itertools
import regex
from tqdm import tqdm
import subprocess
from joblib import Parallel, delayed
import multiprocessing



class Node():
    def __init__(self, id, pose):
        self.id = id
        self.pose = pose
        self.true_pose = pose

    def set_pose(self, pose):
        self.pose = pose

class Edge():
    def __init__(self, vehicle_id, from_id, to_id, covariance, transform, keyframe):
        self.from_id = from_id
        self.to_id = to_id
        self.covariance = covariance
        self.transform = transform
        self.vehicle_id = vehicle_id
        self.KF = keyframe

class Agent():
    def __init__(self, id):
        self.id  = id
        self.loop_closed = False

class Backend():
    def __init__(self, name="default"):
        self.name = name
        self.G = nx.DiGraph()
        self.node_plot_positions = dict()
        self.lc_edges = []
        self.LC_threshold = 1.5
        self.overlap_threshold = 0.75
        self.node_id_map = dict()
        self.agents = []
        self.optimized = True

    def add_agent(self, vehicle_id, KF):
        # Tell the backend to keep track of this agent
        new_agent = Agent(vehicle_id)
        self.agents.append(new_agent)
        self.G.add_node(str(vehicle_id)+"_0", KF=KF)
        if vehicle_id == 0:
            self.agents[0].loop_closed = True

    def add_edge(self, edge):
        # Add this edge to the networkx graph
        # TODO: add checks to make sure we know about this agent
        self.G.add_edge(edge.from_id, edge.to_id, covariance=edge.covariance, transform=edge.transform)

        # Save the keyframe to the node
        self.G.node[edge.to_id]['KF'] = edge.KF

        # Look for loop closures
        node_id, lc_transform, P = self.find_loop_closures(edge.to_id)

        if node_id:
            # Add loop closure edge to networkx graph
            self.G.add_edge(edge.from_id, node_id, covariance=P, transform=lc_transform)


    def find_loop_closures(self, node_id):
        # compare all nodes, and see if the poses are similar
        vehicle_id = int(node_id.split("_")[0])
        node_num = int(node_id.split("_")[1])
        KF_from = np.array(self.G.node[node_id]['KF'])

        for (i, n) in self.G.node.iteritems():
            # don't find loop closure between odometry
            if vehicle_id == int(i.split("_")[0]):
                # If these are from the same vehicle, make sure that these edges are far apart
                if abs(int(i.split("_")[1]) - node_num) < 10:
                    continue

            KF_to = np.array(n['KF'])

            # If this is a loop closure
            if np.linalg.norm(KF_to - KF_from) < self.LC_threshold:
                # If these are different agents, signal that these agents have been loop closed
                if self.agents[int(vehicle_id)].loop_closed or self.agents[int(i.split("_")[0])].loop_closed:
                    self.agents[int(vehicle_id)].loop_closed = True
                    self.agents[int(i.split("_")[0])].loop_closed = True

                # Covariance Matrix for Loop Closure Edge
                P = [[0.0000001, 0, 0],
                     [0, .0000001, 0],
                     [0, 0, .0000001]]

                return i, self.find_transform(KF_from, KF_to), P

        # We didn't find a loop closure
        return 0, 0, 0

    def find_transform(self, from_pose, to_pose):
        # Rotate transform frame to the "from_node" frame
        xij_I = to_pose[0:2] - from_pose[0:2]
        psii = from_pose[2]
        R_I_to_i = np.array([[cos(psii), sin(psii)],
                             [-sin(psii), cos(psii)]])
        dx = xij_I.dot(R_I_to_i.T)
        dpsi = to_pose[2] - psii

        # Pack up and output the loop closure
        return [dx[0], dx[1], dpsi]

    def optimize(self):

        self.plot_graph(self.G)
        plt.show()

        # Get a minimum spanning tree
        graph_undirected = nx.Graph()
        graph_directed = nx.DiGraph()

        for agent in self.agents[1:]:
            if agent.loop_closed:

                ####### I AM HERE ###### 3/23/2017





    def run_g2o(self, graph, initialized):
        g2o_id_map = self.output_g2o(graph, initialized, "edges.g2o")

        # Run g2o
        run_g2o_str = ("../g2o/bin/g2o", "-o", "output.g2o", "-v", "edges.g2o")
        g2o = subprocess.Popen(run_g2o_str, stdout=subprocess.PIPE)
        g2o.wait()

        return self.load_g2o("output.g2o", g2o_id_map)


    def output_g2o(self, graph, initialized, filename):
        node_id_map = dict()
        node_id_map['g2o_index'] = dict()
        node_id_map['vID_index'] = dict()
        f = open(filename, 'w')
        g2o_index = 0

        # Write nodes to file
        for i in graph.nodes_iter():
            node_id_map['g2o_index'][g2o_index] = i
            node_id_map['vID_index'][i] = g2o_index
            # If we have initialized these nodes, then provide the initial guess to g2o
            if initialized:
                line = "VERTEX_SE2 " + str(g2o_index) + " " + \
                    str(self.G.node[i]['pose'][0]) + " " + \
                    str(self.G.node[i]['pose'][1]) + " " + \
                    str(self.G.node[i]['pose'][2]) + "\n"
            # Otherwise, we can't do better than all zeros
            else:
                line = "VERTEX_SE2 " + str(g2o_index) + " " + \
                    str(0) + " " + str(0) + " " + str(0) + "\n"
            f.write(line)
            g2o_index += 1

        # Fix agent 0, node 0 as global origin (Could be moved)
        f.write("FIX " + str(node_id_map['vID_index']['0_0']) + "\n")

        # Write edges to file
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
        f_e = open(g2o_file, 'r')
        nodes = dict()
        graph = nx.DiGraph()
        for line in f_e:
            # Look for the optimized node positions
            if regex.search("VERTEX_SE2", line):
                n_str = line.split(" ")
                node_id = node_id_map['g2o_index'][int(n_str[1])]
                x = float(n_str[2])
                y = float(n_str[3])
                theta = float(n_str[4])
                nodes[node_id] = [x, y, theta]
                graph.add_node(node_id, pose=[x, y, theta], vehicle_id=node_id.split("_")[0])

            # G2O Doesn't modify edges, so we have to re-calculate edges from the optimized
            # node positions
            elif regex.search("EDGE_SE2", line):
                e_str = line.split(" ")
                from_id = node_id_map['g2o_index'][int(e_str[1])]
                to_id = node_id_map['g2o_index'][int(e_str[2])]
                transform = self.find_transform(graph[from_id]['pose'], graph[to_id]['pose'])
                # Total guess about covariance for optimized edges
                P = [[0.0000001, 0, 0],
                     [0, .0000001, 0],
                     [0, 0, .0000001]]
                graph.add_edge(from_id, to_id, transform=transform, covariance=P)
        return graph

    def plot_graph(self, graph, name='default', arrows=True, figure_handle=0, edge_color='m', lc_color='y'):
        if figure_handle:
            plt.figure(figure_handle)
        else:
            plt.figure()
        ax = plt.subplot(111)
        plt.title(name)

        # Get positions of all nodes
        plot_positions = dict()
        for (i, n) in graph.node.iteritems():
            plot_positions[i] = [n['KF'][1], n['KF'][0]]

        nx.draw_networkx(graph, pos=plot_positions,
                         with_labels=True, ax=ax, edge_color=edge_color,
                         linewidths="0.3", node_color='c', node_shape='')
        if arrows:
            for i, n in self.G.node.iteritems():
                pose = n['KF']
                arrow_length = 1.0
                dx = arrow_length * cos(pose[2])
                dy = arrow_length * sin(pose[2])
                # Be sure to convert to NWU for plotting
                ax.arrow(pose[1], pose[0], dy, dx, head_width=arrow_length*0.15, head_length=arrow_length*0.3, fc='c', ec='b')

        # # Plot loop closures
        # for lc in self.lc_edges:
        #     x = [lc[0][0], lc[1][0]]
        #     y = [lc[0][1], lc[1][1]]
        #     self.ax.plot(y, x , lc_color, lw='0.1', zorder=1)

        plt.axis("equal")


