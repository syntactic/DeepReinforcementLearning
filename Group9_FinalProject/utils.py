import numpy as np
import random, os, pickle, torch
from collections import deque
import matplotlib.pyplot as plt

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

        # game play -> [tensors]
        # 
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
            *batch)

        assert len(batch_state) > 0
        assert type(batch_state[0])
        # need to correct this in load_trajectories (?) -> it should match the online samples format
        # for loaded trajectories -> (batch size, 
        # for online samples -> (batch size, 1, 100)
     
        """batch_state = np.array(batch_state)
        batch_next_state = np.array(batch_next_state)
        batch_action = np.array(batch_action)

        print(f"before b {batch_state[0].shape}")
        for i,s in enumerate(batch_state):
            s = format_state(s)
            #batch_state[i] = format_state(s)
        print(f"after b {batch_state[0].shape}")

        print(f"before n_b {batch_next_state[0].shape}")
        for s in batch_next_state:
            s = format_state(s)
        print(f"after n_b {batch_next_state[0].shape}")

        for i in range(len(batch_state)):
            for j in range(len(batch_state[0])):
                print(i, j, len(batch_state[i][j]))

        print('---------------')

        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=device)
        batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float, device=device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.int64, device=device)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
        batch_reward = torch.as_tensor(batch_reward, dtype=torch.float, device=device).unsqueeze(1)
        #print(batch_done)
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device).unsqueeze(1)"""

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

class Model():
    def __init__(self, model, name="model"):
        self.model = model
        self.name = name
        self.loss_bucket = []
        self.losses = []
        self.format_state = lambda x : x
        self.device = 'cpu'
    
    def get_Q(self, states):

        states = [self.format_state(state) for state in states]
        batch_states = np.array(states)
        batch_states = torch.as_tensor(batch_states, dtype=torch.float, device=self.device)

        return self.model(batch_states)
    
    def parameters(self):
        return self.model.parameters()
    
    def append_to_loss_bucket(self, loss_item):
        self.loss_bucket.append(loss_item)

    def plot_losses(self, path = "", save=False):
        # set up the figure and axes
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

        ax.plot(range(len(self.loss_bucket)), self.loss_bucket)
        ax.set_xlabel('training steps')
        ax.set_ylabel('loss')
        
        if save:
            plt.tight_layout()
            fig.savefig(path + self.name + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        plt.show()
    
    def save(self, path=""):
        torch.save(self.model.state_dict(), path+self.name+'.pt')
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def print(self):
        print(self.model)
    
    def estimate_value_map(self, grid, save=False, path=""):
        """ estimates a value map by steping through every available tile in the 
            grid and passing it through the model to get Q values. The value map
            is the max q value for each state """
        
        # set up an empty value map
        V_map = np.empty((grid.width, grid.height))

        # this will update itself as the game progresses
        state = grid.state 

        up_move = True
        down_move = False

        win = False

        while not win:
            #grid.visualize_grid()
            # get current Q val
            with torch.no_grad():
                Q_vals = self.model(self.format_state(state))

            # get max
            V = torch.max(Q_vals, 1).values
            # set value in valuse map
            V_map[grid.player_pos.y, grid.player_pos.x] = V

            # determine if the player needs to turn
            if (up_move and grid.player_pos.y == 0) or (down_move and grid.player_pos.y == grid.height-1):
                grid.step(1)
                up_move = not up_move
                down_move = not down_move
            else:
                if up_move:
                    grid.step(0)
                elif down_move:
                    grid.step(2)
            
            win = grid.check_game_over()
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.imshow(V_map)
        if save:
            plt.tight_layout()
            fig.savefig(path+self.name + "VMap" + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        plt.show()

            
            


class Position:
    def __init__(self, x, y):
        self.x=x
        self.y=y

    def __sub__(self, other_position):
        return Position(self.x - other_position.x, self.y - other_position.y)

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

def get_max_Q(q, alpha=0.001):
    return alpha * torch.logsumexp(q/alpha, dim=1, keepdim=True)

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
