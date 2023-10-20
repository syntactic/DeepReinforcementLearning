import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from Agent import Agent
from GridWorld import GridWorld
from utils import *
import pickle


"""
Basic idea: feed it an agent, a game, and a number of games to play

- it should be able to flexibly allow different agents to interact with 
    the gridworld to play the games

- Data should be saved at the end (this could be within the Agent class, 
    where different child classes of the agent have different implementations 
    of .save())

- learning stuff should be set within the Agent class (?) to me this is more 
    intuitive than having the Orchestrator decide if there should be learning
    or not

- some possible parameters: 
    - fixed/random game boards (or maybe that should be set as part of the GridWorld class)
"""

class Orchestrator:

    def __init__(self, game:GridWorld, agent:Agent, num_games:int):
        self.game = game
        self.agent = agent
        self.num_games = num_games
        self.trajectories = [] # should trajectories belong to agents?
    
    def play(self):

        for g in range(self.num_games):
            print('Playing game', g, '... ', end='')

            # intialize the game
            self.game.reset()
            self.agent.reset() 
            self.trajectories.append([])

            # get initial state and whether game is over
            init_state = self.game.get_state()
            playing = not self.game.check_game_over()

            # set state as init_state
            state = init_state

            # play until the game is over
            while(playing):

                # get action from agent based on state
                action = self.agent.get_action(state)

                # get next_state, reward, and whether the game was won
                next_state, reward, game_over = self.game.step(action)

                # add data to trajectories
                self.trajectories[-1].append((state, action, reward, next_state, game_over))

                # inform the agent of the result
                self.agent.inform_result(next_state, reward, game_over)

                # check whether the game should end
                playing = not game_over

                if not playing:
                    self.agent.save_results()
                    print(self.game.moves_made, " ", end="")
                    print('Game Done.')


    def set_game(self, game):
        """ sets the game controlled by the orchestrator """
        self.game = game
    
    def set_agent(self, agent):
        """ sets the agent controlled by the orchestrator """
        self.agent = agent

    def save_trajectories(self, filepath='trajectories.pkl'):
        with open(filepath, 'wb') as file:
            states, actions, rewards, next_states, dones = split_trajectories(self.trajectories)
            pickle.dump({'states':states,
                         'actions':actions,
                         'rewards':rewards,
                         'next_states':next_states,
                         'dones':dones}, file)
