from backend import *
from robot import *

def wandering_input(time):
    u = []
    dt = time[1] - time[0]
    for i in range(0, 4):
        u += [[1.0, 0.0 + np.random.normal(0, 0.2)] for t in range(0, (len(time)/5 - 20))]
        u+= [[0.25, pi/(2*20*dt) + np.random.normal(0, 0.2)] for t in range(0, 20)]
    u += [[1.0, 0.0] for t in range(0, (len(time) / 5 - 20))]
    return u

dt = 0.01
time = np.arange(0, 10.01, dt)

Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
model = Robot(0, 0, 0, Q)
input = wandering_input(time)

for t, u in zip(time, input):
    model.propagate_dynamics(u, dt)
    if t % 0.25 == 0 and t > 0:
        model.reset()

for edge in model.edges:
    print edge

global_state = model.find_global_state()
# model.draw_trajectory()
debug = 1

n1 = Node(1, [0, 0, 0])
n2 = Node(2, [1, 0, 1.507])
n3 = Node(3, [1, 1, 3.14159])
n4 = Node(4, [0, 1, -1.507])

P = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
e1 = Edge(0, 1, P, [1, 0, 0.707])
e2 = Edge(1, 2, P, [1, 0, 0.707])
e3 = Edge(2, 3, P, [1, 0, 0.707])
e4 = Edge(3, 4, P, [1, 0, 0.707])

map = Backend()

# map.add_node(n1)
# map.add_node(n2)
# map.add_node(n3)
# map.add_node(n4)

map.add_edge(e1)
map.add_edge(e2)
map.add_edge(e3)
map.add_edge(e4)

map.plot_graph()