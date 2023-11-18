import numpy as np
from utils import get_max_Q, index_of_value_in_2d_array, Position
from plotting_utils import plot_values_over_index, plot_heatmap, save_fig
import matplotlib.pyplot as plt
import torch
from constants import *
from GridWorld import GridWorld

class Model():
    """The Model class is a wrapper for a PyTorch model. It contains the PyTorch
    model itself, a preprocessing function, and a loss tracker. It also contains
    several helper functions to help assess what the model learned. """

    def __init__(self, model, name="model"):
        self.model = model
        self.name = name
        self.loss_bucket = []
        self.format_state = lambda x : x
        self.device = 'cpu'
    
    def get_Q(self, states, no_grad=False):
        """ Preprocesses batches of states and returns Q values """

        states = [self.format_state(state) for state in states]
        batch_states = np.array(states)
        batch_states = torch.as_tensor(batch_states, dtype=torch.float, device=self.device)

        if no_grad:
            with torch.no_grad():
                return self.model(batch_states)
        return self.model(batch_states)
    
    def parameters(self):
        return self.model.parameters()
    
    def append_to_loss_bucket(self, loss_item):
        self.loss_bucket.append(loss_item)

    def plot_losses(self, path = "", save=False):
        plot_values_over_index(self.loss_bucket, path=path, filename=self.name + "_losses",
                xlabel='training steps', ylabel='loss', save=save)

    def save(self, path=""):
        torch.save(self.model.state_dict(), path+self.name+'.pt')
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def print(self):
        print(self.model)

    def estimate_reward_map(self, grid, save=False, path=""):
        """ Estimates rewards by running through games and deducing rewards 
        from the Q values of a trained model. Reward for a state is the expected
        value of reward sampling actions to that state from a policy. """
        reward_map = np.zeros((grid.width, grid.height))
        reward_bucket = [[[] for i in range(grid.width)] for j in range(grid.height)]

        # use this to explictly init the player at each position so there is 
        # at least one sample for every part of the grid
        state_list = grid.get_full_state_space()
        gamma = 0.9
        
        i = 0
        while i < 100:
            for j in range(len(state_list)): # sample a thousand times?
                # get player position
                (y, x) = index_of_value_in_2d_array(state_list[j], PLAYER)
                grid.reset()
                grid.player_pos = Position(x, y)
                grid.generate_grid()
                # get initial playing boolean 
                playing = not grid.check_game_over()
                while playing:
                    state = grid.get_state()
                    Q_vals = self.get_Q([state], no_grad=True).squeeze()
                    action = grid.action_space[np.argmax(Q_vals)]
                    current_Q = Q_vals[action]
                    next_state, _, game_over = grid.step(action)
                    next_V = get_max_Q(self.get_Q([next_state], no_grad=True))
                    y = (1 - game_over) * gamma * next_V
                    reward = current_Q - y
                    row, col = index_of_value_in_2d_array(next_state, PLAYER)
                    reward_bucket[row][col].append(reward)

                    playing = not game_over
            i += 1
        for row in range(grid.height):
            for col in range(grid.width):
                if len(reward_bucket[row][col]) > 0:
                    reward_map[row, col] = np.mean(reward_bucket[row][col])

        plot_heatmap(data=reward_map, path=path+self.name + "_reward_map", save=save)
        np.savetxt(path+self.name + "_reward_map.txt", reward_map, fmt='%8.5f')

    def estimate_value_map(self,grid,save=False, path=""):
        """ estimates a value map by steping through every available tile in the 
            grid and passing it through the model to get Q values. The value map
            is the max q value for each state """
        
        # set up an empty value map
        V_map = np.empty((grid.width, grid.height))
        actions_map = np.empty((grid.width, grid.height), dtype=int)

        # get state list
        state_list = grid.get_full_state_space()
        for s in state_list:

            # get V for current state
            with torch.no_grad():
                Q_vals = self.model(self.format_state(s))
            (V, index) = torch.max(Q_vals, 1)

            # index into value map
            V_map[s==PLAYER] = V
            actions_map[s==PLAYER] = int(index[0])
        
        plot_heatmap(data=V_map, path=path+self.name + "_VMap", save=save)
        np.savetxt(path+self.name + "_Vmap.txt", V_map, fmt='%8.3f')

        return V_map, actions_map

    def plot_argmax_policy(self, policy_map, save=False, path=""):
        """ plots the argmax policy of the model 
            (action that leads to the max Q state)"""
        
        # create quiver plot of arrows to specify the actions 
        dx = [np.zeros((policy_map.shape[0],1))]
        dy = [np.zeros((policy_map.shape[1],1))]

        dx_map = {0:0, 1:1, 2:0, 3:-1}
        dy_map = {0:1, 1:0, 2:-1, 3:0}
        cell_origin_map = {0:(0,0.4), 1:(-0.4,0), 2:(0,-0.4), 3:(0.4,0)}

        fig, ax = plt.subplots(figsize = (7, 7))
        ax.imshow(policy_map)
        plt.axis('off')

        for x in range(policy_map.shape[0]):
            for y in range(policy_map.shape[1]):
                curr_action = policy_map[y,x]
                ax.quiver(x + cell_origin_map[curr_action][0],
                          y + cell_origin_map[curr_action][1],
                          dx_map[curr_action], 
                          dy_map[curr_action],
                          scale=1.3,
                          scale_units='xy')
        
        if save:
            save_fig(fig, path=path+self.name+"_policy")

        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()
