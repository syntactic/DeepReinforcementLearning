import numpy as np
from utils import plot_values_over_index
import matplotlib.pyplot as plt
import torch
from constants import *
from GridWorld import GridWorld

class Model():
    def __init__(self, model, name="model"):
        self.model = model
        self.name = name
        self.loss_bucket = []
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
        plot_values_over_index(self.loss_bucket, filename=self.name + "_losses",
                xlabel='training steps', ylabel='loss', save=save)

    def save(self, path=""):
        torch.save(self.model.state_dict(), path+self.name+'.pt')
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def print(self):
        print(self.model)

    def estimate_reward_map(self, grid, save=False, path=""):
        reward_map = np.empty((grid.width, grid.height))
        state_list = grid.get_full_state_space()
        gamma = 0.9

        for next_state in state_list:
            rewards_for_next_state = []
            for action in ACTIONS:
                g_next_state = GridWorld.from_state(next_state, grid.win_state)
                prev_state, _, _ = g_next_state.step(action)
                if np.array_equal(prev_state, next_state):
                    continue
                action_reverse = (action + 2) + (action >= 2) * -4
                done = g_next_state.player_pos == g_next_state.win_state
                Q_vals = self.get_Q(prev_state)
                current_Q = Q_vals.gather(2, action).squeeze()
                next_V = get_max_Q(self.model.get_Q(next_state))
                y = (1 - done) * gamma * next_V
                reward = current_Q - y
                rewards_for_next_state.append(reward)
            if len(rewards_for_next_state) > 0:
                reward_map[index_of_value_in_2d_array(next_state, PLAYER)] = np.mean(rewards_for_next_state)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.imshow(reward_map)
        if save:
            plt.tight_layout()
            fig.savefig(path+self.name + "reward_map" + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        plt.show()

    
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
        
        V_map[grid.win_state.y, grid.win_state.x] = 0.0
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.imshow(V_map)
        if save:
            plt.tight_layout()
            fig.savefig(path+self.name + "VMap" + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        plt.show()
