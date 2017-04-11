#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from math import *
import numpy as np
import regex
from tqdm import tqdm
import subprocess
import scipy.spatial


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
        self.G = nx.Graph()
        self.node_plot_positions = dict()
        self.lc_edges = []
        self.LC_threshold = 1.5
        self.overlap_threshold = 0.75
        self.node_id_map = dict()
        self.agents = []
        self.optimized = True

        self.keyframe_map = dict()
        self.keyframes = []
        self.current_keyframe_index = 0


    def add_keyframe(self, KF, node_id):
        # Add the keyframe to the map
        self.keyframes.append(KF)
        self.keyframe_map[self.current_keyframe_index] = node_id
        self.current_keyframe_index += 1


    def add_agent(self, vehicle_id, KF):
        # Tell the backend to keep track of this agent
        new_agent = Agent(vehicle_id)
        self.agents.append(new_agent)
        self.G.add_node(str(vehicle_id)+"_000", KF=KF)

        # Add keyframe to the map
        self.add_keyframe(KF, str(vehicle_id)+"_000")

        if vehicle_id == 0:
            self.agents[0].loop_closed = True


    def add_edge(self, edge):
        # Add this edge to the networkx graph
        # TODO: add checks to make sure we know about this agent
        self.G.add_edge(edge.from_id, edge.to_id, covariance=edge.covariance, transform=edge.transform,
                        from_id=edge.from_id, to_id=edge.to_id)
        # Save the keyframe to the node
        self.G.node[edge.to_id]['KF'] = edge.KF
        #Add Keyframe to the map
        self.add_keyframe(edge.KF, edge.to_id)



    def find_loop_closures(self):
        # Build a KDtree to search
        tree = scipy.spatial.KDTree(self.keyframes)
        lc_count = 0

        print("finding loop closures")
        for from_id in tqdm(self.G.node):
            KF_from = self.G.node[from_id]['KF']
            indices = tree.query_ball_point(KF_from, self.LC_threshold, 2.0)
            for index in indices:
                if abs(self.current_keyframe_index - index) > 10:
                    to_id = self.keyframe_map[index]
                    P = [[0.001, 0, 0], [0, .001, 0], [0, 0, .001]]
                    KF_to = self.keyframes[index]
                    self.G.add_edge(from_id, to_id, covariance=P,
                                    transform=self.find_transform(np.array(KF_from), np.array(KF_to)),
                                    from_id=from_id, to_id=to_id)
                    lc_count += 1
        print("found %d loop closures" % lc_count)


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

    def find_pose(self, graph, node, origin_node):
        if 'pose' in graph.node[node]:
            return graph.node[node]['pose']
        else:
            path_to_origin = nx.shortest_path(graph, node, origin_node)
            edge = graph.edge[node][path_to_origin[1]]
            # find the pose of the next closest node (this is recursive)
            nearest_known_pose = self.find_pose(graph, path_to_origin[1], origin_node)

            # edges could be going in either direction
            # if the edge is pointing to this node
            if edge['from_id'] == path_to_origin[1]:
                psi0 = nearest_known_pose[2]
                x = nearest_known_pose[0] + edge['transform'][0] * cos(psi0) - edge['transform'][1] * sin(psi0)
                y = nearest_known_pose[1] + edge['transform'][0] * sin(psi0) + edge['transform'][1] * cos(psi0)
                psi = psi0 + edge['transform'][2]
            else:
                psi = nearest_known_pose[2] - edge['transform'][2]
                x = nearest_known_pose[0] - edge['transform'][0] * cos(psi) + edge['transform'][1] * sin(psi)
                y = nearest_known_pose[1] - edge['transform'][0] * sin(psi) - edge['transform'][1] * cos(psi)

            graph.node[node]['pose'] = [x, y, psi]
            return [x, y, psi]

    def optimize(self):
        # Find loop closures
        self.find_loop_closures()

        # plt.ion()
        self.plot_graph(self.G, "full graph (TRUTH)")

        # Get a minimum spanning tree of nodes connected to our origin node
        min_spanning_tree = nx.Graph()
        for component in sorted(nx.connected_component_subgraphs(self.G, copy=True), key=len, reverse=True):
            if '0_000' in component.node:
                self.plot_graph(component, "connected component truth", figure_handle=2, edge_color='r')
                # Find Initial Guess for node positions
                component.node['0_000']['pose'] = [0, 0, 0]
                print("seeding initial graph")
                for node in tqdm(component.nodes()):
                    self.find_pose(component, node, '0_000')
                self.plot_graph(component, "connected component unoptimized", figure_handle=3, edge_color='m')

                # Let GTSAM crunch it
                print("optimizing")
                optimized_component = self.run_g2o(component)
                self.plot_graph(optimized_component, "optimized", figure_handle=4, edge_color='b')

                debug = 1
        plt.show()

        debug = 1









    def run_g2o(self, graph):
        self.output_g2o(graph, "edges.g2o")

        # Run g2o
        run_g2o_str = "../jax_optimizer/build/jax_optimizer"
        g2o = subprocess.Popen(run_g2o_str)
        g2o.wait()

        return self.load_g2o("output.g2o")


    def output_g2o(self, graph, filename):
        f = open(filename, 'w')
        g2o_index = 0

        # Write nodes to file
        for i in sorted(graph.nodes_iter()):
            if 'pose' not in graph.node[i]:
                error = 1
            line = "VERTEX_SE2 " + i + " " + \
                str(graph.node[i]['pose'][0]) + " " + \
                str(graph.node[i]['pose'][1]) + " " + \
                str(graph.node[i]['pose'][2]) + "\n"
            f.write(line)
            g2o_index += 1

        # Fix agent 0, node 0 as global origin (Could be moved)
        f.write("FIX 0_0000\n")

        # Write edges to file
        for pair in graph.edges():
            i = pair[0]
            j = pair[1]
            edge = graph.edge[i][j]
            line = "EDGE_SE2 " + edge['from_id'] + \
                    " " + edge['to_id'] + \
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

    def load_g2o(self, g2o_file):
        f_e = open(g2o_file, 'r')
        nodes = dict()
        graph = nx.Graph()
        for line in f_e:
            # Look for the optimized node positions
            if regex.search("VERTEX_SE2", line):
                n_str = line.split(" ")
                node_id = n_str[1]
                x = float(n_str[2])
                y = float(n_str[3])
                theta = float(n_str[4])
                nodes[node_id] = [x, y, theta]
                graph.add_node(node_id, pose=[x, y, theta], vehicle_id=node_id.split("_")[0],
                               KF=self.G.node[node_id]['KF'])

            # G2O Doesn't modify edges, so we have to re-calculate edges from the optimized
            # node positions
            elif regex.search("EDGE_SE2", line):
                line = line.rstrip('\n')
                e_str = line.split(" ")
                from_id = e_str[1]
                to_id = e_str[2]
                transform = self.find_transform(np.array(graph.node[from_id]['pose']),
                                                np.array(graph.node[to_id]['pose']))
                # Total guess about covariance for optimized edges
                P = [[0.00001, 0, 0],
                     [0, 0.00001, 0],
                     [0, 0, 0.00001]]
                graph.add_edge(from_id, to_id, transform=transform, covariance=P)
        return graph

    def plot_graph(self, graph, title='default', name='default', arrows=False, figure_handle=0, edge_color='m', lc_color='y'):
        if figure_handle:
            plt.figure(figure_handle)
        else:
            plt.figure()
            plt.clf()
        plt.title(title)
        ax = plt.subplot(111)

        # Get positions of all nodes
        plot_positions = dict()
        for (i, n) in graph.node.iteritems():
            if 'pose' in n:
                plot_positions[i] = [n['pose'][1], n['pose'][0]]
            else:
                plot_positions[i] = [n['KF'][1], n['KF'][0]]

        nx.draw_networkx(graph, pos=plot_positions,
                         with_labels=False, ax=ax, edge_color=edge_color,
                         linewidths="0.3", node_color='c', node_shape='')
        if arrows:
            for i, n in graph.node.iteritems():
                if 'pose' in n:
                    pose = n['pose']
                else:
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


