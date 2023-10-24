import numpy as np
import random, os, pickle, torch
from collections import deque

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

        batch_state = np.array(batch_state)
        batch_next_state = np.array(batch_next_state)
        batch_action = np.array(batch_action)

        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=device)
        batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float, device=device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.int64, device=device)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
        batch_reward = torch.as_tensor(batch_reward, dtype=torch.float, device=device).unsqueeze(1)
        #print(batch_done)
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device).unsqueeze(1)

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

            idx = perm[:num_trajectories]
            for i in idx:
                self.add((trajs["states"][i], trajs["actions"][i], trajs["rewards"][i],
                    trajs["next_states"][i], trajs["dones"][i]))
        else:
            raise ValueError(f"{filepath} is not a valid path")

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

def read_file(path: str, file_handle):
    if path.endswith("pkl"):
        data = pickle.load(file_handle)
    else:
        raise NotImplementedError
    return data
