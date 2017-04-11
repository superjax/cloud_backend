from backend import *
from robot import *
from controller import *
from tqdm import tqdm


dt = 0.1
time = np.arange(0, 60.01, dt)

robots = []
controllers = []
num_robots = 300
KF_frequency_s = 1.0

map = Backend("Noisy Map")
true_map = Backend("True Map")

start_pose_range = [75, 75, 6.28318530718]


start_poses = np.array([np.random.uniform(-start_pose_range[i],
                                 start_pose_range[i],
                                 num_robots).tolist()
               for i in range(3)]).T.tolist()
start_poses[0] = [0, 0, 0]

P_perfect = [[0.00001, 0, 0], [0, 0.00001, 0], [0, 0, 0.00001]]
G = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
print("simulating robots")
for r in tqdm(range(num_robots)):
    # Run each robot through the trajectory
    robots.append(Robot(r, G, start_poses[r]))
    controllers.append(Controller())
    control = [controllers[r].control(t) for t in time]
    for t, u in zip(time, control):
        robots[r].propagate_dynamics(u, dt)
        if t % KF_frequency_s == 0 and t > 0:
            robots[r].reset()

    # robots[r].draw_trajectory()

    # Put edges in backends
    i = 0
    map.add_agent(r, KF=start_poses[r])
    for edge, KF in zip(robots[r].edges, robots[r].keyframes):
        e = Edge(r, str(r) + "_" + str(i).zfill(3), str(r) + "_" + str(i+1).zfill(3), G, edge, KF)
        map.add_edge(e)
        i += 1

    i = 0
    true_map.add_agent(r, KF=start_poses[r])
    for edge, KF in zip(robots[r].true_edges, robots[r].keyframes):
        e = Edge(r, str(r) + "_" + str(i).zfill(3), str(r) + "_" + str(i+1).zfill(3), P_perfect, edge, KF)
        true_map.add_edge(e)
        i += 1

# Find loop closures
# print("finding loop closures")
# loop_closures = true_map.simulate_loop_closures()
# print("found %d loop closures" % len(loop_closures))
# for lc in loop_closures:
#     map.add_edge(lc)

# print("plotting truth")
# true_map.plot_graph(figure_handle=1, edge_color='r', lc_color='y', arrows=True)
# print("plotting estimates")
# map.plot_graph(figure_handle=1, edge_color='g', lc_color='m', arrows=True)
# # print("plotting optimized")
# # optimized_map.plot_graph(figure_handle=1, edge_color='b', lc_color='m', arrows=True)
# plt.show()

map.optimize()