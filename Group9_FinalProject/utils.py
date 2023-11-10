import numpy as np
import random, os, pickle, torch
from collections import deque
import matplotlib.pyplot as plt
from constants import *

class Buffer():
    # heavily inspired by https://github.com/Div99/IQ-Learn/iq_learn/dataset/*
    def __init__(self, memory_size: int = 512, seed: int = 0) -> None:
        random.seed(seed)
        self.memory_size = memory_size
        self.data = deque(maxlen=self.memory_size)

    # (s, a, r, s', d)
    def add(self, timestep_data) -> None:
        #print(f"timestep_data: {timestep_data}")
        self.data.append(timestep_data)

    def size(self):
        return len(self.data)

    def sample(self, batch_size: int):
        if batch_size > len(self.data):
            batch_size = len(self.data)
        indexes = np.random.choice(np.arange(len(self.data)), size=batch_size, replace=False)
        return [self.data[i] for i in indexes]

    def get_samples(self, batch_size, device='cpu'):
        batch = self.sample(batch_size)

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
            *batch)

        assert len(batch_state) > 0

        return batch_state, batch_next_state, batch_action, batch_reward, batch_done

    def load_trajectories(self, filepath: str,
                          num_trajectories: int = 10,
                          seed: int = 0):
        if os.path.isfile(filepath):
            # Load data from single file.
            with open(filepath, 'rb') as f:
                trajs = read_file(filepath, f)

            rng = np.random.RandomState(seed)
            # Sample random `num_trajectories` experts.
            perm = np.arange(len(trajs["states"]))
            perm = rng.permutation(perm)

            # idx refers to each trajectory
            idx = perm[:num_trajectories] 

            # for each trajectory
            for i in idx:
                for j in range(len(trajs["states"][i])):
                    self.add((trajs["states"][i][j], trajs["actions"][i][j], trajs["rewards"][i][j],
                        trajs["next_states"][i][j], trajs["dones"][i][j]))
        else:
            raise ValueError(f"{filepath} is not a valid path")

class Position:
    def __init__(self, x, y):
        self.x=x
        self.y=y

    def __sub__(self, other_position):
        return Position(self.x - other_position.x, self.y - other_position.y)

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __repr__(self):
        return f"({self.x}, {self.y})"
        

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

def read_file(path: str, file_handle):
    if path.endswith("pkl"):
        data = pickle.load(file_handle)
    else:
        raise NotImplementedError
    return data

def get_max_Q(Q, alpha=0.001):
    return (alpha * torch.logsumexp(Q/alpha, dim=2, keepdim=True)).squeeze()

# loss(input, output) -> iq_loss(current_V, y)
def iq_loss(current_Q, current_V, y): # args, etc
    """ heavily inspired by https://github.com/Div99/IQ-Learn/blob/main/iq_learn/iq.py """
    # Notes: (our explanation of what iq loss is doing)
    # the loss takes in 2 points -> 
    #       'reward' (calculated as current Q - expected value of next state)
    #       'value_loss' (calculated as current V - expected value of next state)
    # loss tries to minimize the difference between reward and value_loss

    #  calculate 1st term for IQ loss
    #  -E_(ρ_expert)[Q(s, a) - γV(s')]
    #y = (1 - done) * gamma * next_V    curr_Qs = [0.1, 0.1, 0.3, (0.2)], next_Qs = [0.2, 0.1, 0.05, 0.03]
    reward = (current_Q - y) # [is_expert]  ## ( 0.2 - 0.99*0.5) -> NEG

    # phi_grad is the divergence term
    phi_grad = 1
    loss = -(phi_grad * reward).mean() # then, loss is POS

    # sample using only expert states (works offline)
    value_loss = (current_V - y).mean() # (0.3 - 0.99 * 0.2) -> POS > reward
    loss += value_loss # (more POS) (further from 0) :()

    return loss

def get_concat_samples(policy_batch, expert_batch):
    """ borrowed from https://github.com/Div99/IQ-Learn/blob/main/iq_learn/utils/utils.py """
    online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = policy_batch

    expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = expert_batch

    batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
    batch_next_state = torch.cat(
        [online_batch_next_state, expert_batch_next_state], dim=0)
    batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
    batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
    batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)
    is_expert = torch.cat([torch.zeros_like(online_batch_reward, dtype=torch.bool),
                           torch.ones_like(expert_batch_reward, dtype=torch.bool)], dim=0)

    return batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert

def plot_values_over_index(data, path = "", filename="graph", xlabel='x', ylabel='y', figsize=(8, 4), save=False):
    # set up the figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    ax.plot(range(len(data)), data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save:
        plt.tight_layout()
        fig.savefig(path + filename + '.svg', format='svg', dpi=1200, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()

def index_of_value_in_2d_array(arr : np.ndarray, val):
    val_index = np.where(arr == val)
    if val_index[0].size == 0:
        return -1
    return val_index[0][0], val_index[1][0]


# put these here for now
def unroll_grid(state):
    if torch.is_tensor(state):
        state = state.numpy()
    w, h = state.shape
    
    s = state.reshape((1, w*h )) + \
        np.random.rand(1, w*h)/10.0 
    s = torch.from_numpy(s).float() 
    return s

def init_grid_model(input_size, action_space):
    """ provides an default model for the gridworld problem """
    l1 = input_size
    l2 = 150
    l3 = 100
    l4 = len(action_space)

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3,l4)
    )

    return model
