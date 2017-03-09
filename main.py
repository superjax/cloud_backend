from backend import *
from robot import *
from controller import *

dt = 0.01
time = np.arange(0, 600.01, dt)

Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
model = Robot(0, 0, 0, Q)
controller = Controller()
input = [controller.control(t) for t in time]

for t, u in zip(time, input):
    model.propagate_dynamics(u, dt)
    if t % 1.0 == 0 and t > 0:
        model.reset()

for edge in model.edges:
    print edge

global_state = model.find_global_state()
# model.draw_trajectory()
debug = 1

map = Backend()

i = 0
P = [[0.1, 0, 0],
     [0, 0.1, 0],
     [0, 0, 0.1]]
for edge in model.edges:
    e = Edge(i, i+1, P, edge)
    map.add_edge(e)
    i += 1

map.plot_graph()