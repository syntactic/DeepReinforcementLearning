import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime

from GridWorld import GridWorld
from Agent import Agent

class DQN:
    def __init__(self, game, num_epochs=100, gamma=0.9, max_moves_per_game=100):
        self.game = game
        self.model = self.init_grid_model()
        self.loss_fn = torch.nn.MSELoss()
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.gamma = gamma
        self.epsilon = 1.0
        self.num_epochs = num_epochs
        self.losses = np.empty((self.num_epochs, 1))
        self.rewards = np.empty((self.num_epochs, 1))
        self.max_moves_per_game = max_moves_per_game

    def init_grid_model(self):
        """ provides an default model for the gridworld problem """
        l1 = self.game.num_states
        l2 = 150
        l3 = 100
        l4 = len(self.game.action_space)
 
        model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3,l4)
        )

        return model
    
    def train(self):
        """ train model for self.num_epochs games of gridworld """
        for e in range(self.num_epochs):
            print('epoch', e, '... ', end=" ")

            # reset the game to start at the beginning
            self.game.reset(self.game.seed) 

            # get the initial state of the board and add a little random noise
            init_state = self.game.state.reshape((1, self.game.num_states)) + \
                np.random.rand(1,self.game.num_states)/10.0 
            
            # sets the current state to the correct format for inputting to the model
            state = torch.from_numpy(init_state).float() 

            playing = True
            moves = 0

            # track game rewards 
            game_rewards = np.empty((self.max_moves_per_game,1))

            while(playing):
                
                # get qval from the model
                qval_tensor = self.model(state)
                qval = qval_tensor.data.numpy() 
                

                # choose the next action (random or on policy based on epsilon)
                    # currently, the agent can choose impossible actions, which
                    # will result in no change to the state
                if (random.random() < self.epsilon):
                    action_idx = np.random.randint(0,len(self.game.action_space))
                else:
                    action_idx = np.argmax(qval)
                
                action = self.game.action_space[action_idx]

                # take the action
                [next_state, reward, game_won] = self.game.step(action)
                
                # reformat the next state so that it can be inputted into the model
                next_state = next_state.reshape((1, self.game.num_states)) + \
                    np.random.rand(1, self.game.num_states)/10.0
                
                next_state = torch.from_numpy(next_state).float() 

                with torch.no_grad():
                    newQ = self.model(next_state)
                maxQ = torch.max(newQ) 

                # calculate reward with decay 
                if not game_won:                           
                    Y = reward + (self.gamma * maxQ)
                else:
                    Y = reward
                
                
                # append current reward to list
                game_rewards[moves] = reward


                output = torch.Tensor([Y]).detach()
                input = qval_tensor.squeeze()[action_idx]

                loss = self.loss_fn(input, output)
                self.optimizer.zero_grad()
                loss.backward()
                self.losses[e] = loss.item() # change this collect losses over the game
                self.optimizer.step()

                # update the state to the next_state value
                state = next_state
                moves +=1

                # terminate loop if game was won
                if(game_won or moves == self.max_moves_per_game):
                    playing = False
                    print('done')
                
                # epsilon decay
                if self.epsilon > 0.1:
                    self.epsilon -= 1.0/self.num_epochs

            # record the current training step reward
            self.rewards[e] = np.nanmean(game_rewards)

        self.save_model()

    def test(self, num_games = 10, max_moves = 30):
        """ 
        tests the model with num_games games as returns trajectories and 
        rewards for each

        returns the game seeds and a (num_games, max_moves, 2) matrix of 
        game data with the agent pos_x and pos_y for each move in each game
        """
        results = np.empty((num_games, max_moves, 2))
        game_seeds = np.empty((num_games,1))
        rewards = np.empty((num_games,1))

        for g in range(num_games):

            # reset the game to start at the beginning
            self.game.reset(self.game.seed)
            game_seeds[g] = self.game.seed

            # make container for game rewards
            game_rewards = np.empty((self.max_moves_per_game,1))

            # get the initial state of the board and add a little random noise
            init_state = self.game.state.reshape((1, self.game.num_states)) + \
                np.random.rand(1,self.game.num_states)/10.0 
            
            # sets the current state to the correct format for inputting to the model
            state = torch.from_numpy(init_state).float() 

            move = 0
            playing = True
            while (playing):

                # get qval from the model
                qval_tensor = self.model(state)
                qval = qval_tensor.data.numpy() 

                # choose the next action (random or on policy based on epsilon)
                    # currently, the agent can choose impossible actions, which
                    # will result in no change to the state
                
                action_idx = np.argmax(qval)
                #action_idx = np.random.randint(0,len(self.game.action_space))

                action = self.game.action_space[action_idx]

                # take the action
                [next_state, reward, game_won] = self.game.step(action)

                # record current agent position and reward
                results[g, move, 0] = self.game.player_pos.x
                results[g, move, 1] = self.game.player_pos.y
                game_rewards[move] = reward

                if game_won or move == max_moves - 1:
                    playing = False
                    # results (num_games, max_moves, 2(x,y)) 
                    results[g,move:max_moves,0] = self.game.player_pos.x
                    results[g,move:max_moves,1] = self.game.player_pos.y
                
                move +=1

                # reformat the next state so that it can be inputted into the model
                next_state = next_state.reshape((1, self.game.num_states)) + \
                    np.random.rand(1, self.game.num_states)/10.0
                
                next_state = torch.from_numpy(next_state).float() 

                # update the state to the next_state value
                state = next_state
            rewards[g] = np.nanmean(game_rewards)

        return [game_seeds, rewards, results] 
     
    def plot_training_results(self, save=False, path="", name=""):
        """ plots losses over epochs, optionally save as png """

        # set up the figure and axes
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 8))

        axes[0].plot(range(self.num_epochs), self.losses)
        axes[0].set_xlabel('epochs')
        axes[0].set_ylabel('loss')

        axes[1].plot(range(self.num_epochs), self.rewards)
        axes[1].set_xlabel('epochs')
        axes[1].set_ylabel('reward')

        if save:
            plt.savefig(path + name + '.png', bbox_inches='tight')

        plt.show()

    def plot_test_results(self, game_seeds, rewards, results):
        """ 
        plots the results of a test, including trajectories 
        and average reward per game

        """

        ### The idea is for later to loop through games with potentially different seeds
        ### in an animated plot, but right now the games are always intialized with the 
        ### same seed

        # set up the figure and axes
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 8))

        axes[1].set_ylabel('avg. reward')
        axes[1].set_xlim(-3, 4)

        # plot game board in the background
        self.game.reset(self.game.seed)
        # discrete color map to interpret the grid: floor, wall, agent, win
        cmap = colors.ListedColormap(['#CACACA', '#625151', '#E53A3A', '#FFE400'])
        axes[0].imshow(self.game.state, cmap=cmap)
        
        # make small plotting offset for each game trajectory so the plots do not overlap
        offset = np.linspace(-0.25, 0.25, len(game_seeds))

        for game in range(results.shape[0]):
            game_results = results[game,:,:]

            # plot the agent's trajectory
            xs = game_results[:,0] + offset[game]
            ys = game_results[:,1] + offset[game]
            axes[0].plot(xs, ys)

            # plot the average reward
            axes[1].scatter([random.random()], rewards[game])

        plt.show()
        

    def save_model(self, path = "", name=""):
        ##### NOT CURRENTLY WORKING AS EXPECTED
        ## https://pytorch.org/tutorials/beginner/saving_loading_models.html 
        ## (At least the way I am loading it, so check this out)

        """ save the model as a pickle so that it can loaded in the future """
        if name == "":
            PATH = path + 'test.pickle'
        else:
            PATH = path + name + '.pickle'
        torch.save(self.model.state_dict(), PATH)
    
    def set_model(self, model):
        """ set the model of the dqn as model argument"""
        self.model = model
