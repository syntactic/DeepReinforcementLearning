import numpy as np

class Position:
    def __init__(self, x, y):
        self.x=x
        self.y=y

# traj = [[s, a, r, s'], [s, a, r, s'], ...]
# trajectories = [[[s, a, r, s'], [s, a, r, s'], ...], [...], [...], ...]
def split_trajectories(trajectories):
    num_trajectories = len(trajectories)
    states = [[step[0] for step in trajectories[i]] for i in range(num_trajectories)]
    actions = [[step[1] for step in trajectories[i]] for i in range(num_trajectories)]
    rewards = [[step[2] for step in trajectories[i]] for i in range(num_trajectories)]
    next_states = [[step[3] for step in trajectories[i]] for i in range(num_trajectories)]
    dones = [[step[4] for step in trajectories[i]] for i in range(num_trajectories)]

    return states, actions, rewards, next_states, dones
