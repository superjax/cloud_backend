from backend import *
from robot import *
from controller import *

import subprocess


dt = 0.01
time = np.arange(0, 600.01, dt)

Q = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
model = Robot(0, 0, 0, Q)
controller = Controller()
input = [controller.control(t) for t in time]

for t, u in zip(time, input):
    model.propagate_dynamics(u, dt)
    if t % 1.0 == 0 and t > 0:
        model.reset()

global_state = model.find_global_state()
# model.draw_trajectory()
debug = 1

map = Backend("Noisy Map")
true_map = Backend("True Map")

i = 0
for edge in model.edges:
    e = Edge(i, i+1, Q, edge)
    map.add_edge(e)
    i += 1

i = 0
P = [[0.00001, 0, 0],
     [0, 0.00001, 0],
     [0, 0, 0.00001]]
for edge in model.true_edges:
    e = Edge(i, i+1, P, edge)
    true_map.add_edge(e)
    i += 1

# Find loop closures
loop_closures = true_map.simulate_loop_closures()
for lc in loop_closures:
    map.add_edge(lc)

# Smash through g2o
map.output_g2o("edges.g2o")
true_map.output_g2o("truth.g2o")

# Run g2o
subprocess.Popen("pwd")
run_g2o = ("../g2o/bin/g2o", "-o", "output.g2o", "-v", "edges.g2o")
g2o = subprocess.Popen(run_g2o, stdout=subprocess.PIPE)
g2o.wait()

print("loading g2o file")
optimized_map = Backend("Optimized Map")
optimized_map.load_g2o("output.g2o")

print("plotting truth")
true_map.plot_graph(figure_handle=1, edge_color='r', lc_color='y', arrows=True)
print("plotting estimates")
map.plot_graph(figure_handle=1, edge_color='g', lc_color='m', arrows=True)
print("plotting optimized")
optimized_map.plot_graph(figure_handle=1, edge_color='b', lc_color='m', arrows=True)
plt.show()