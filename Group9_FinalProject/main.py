import numpy as np
import matplotlib.pyplot as plt

from GridWorld import GridWorld
from Agent import Agent
from DQNAgent import DQNAgent
from IQLearnAgent import IQLearnAgent
from HumanAgent import HumanAgent
from Orchestrator import Orchestrator
from utils import *

import torch

RANDOM_AGENT = 0
DQN_AGENT = 1
HUMAN_AGENT = 2
IQ_LEARN_AGENT = 3

STATIC_START = 0
RANDOM_START = 1

STATIC_WALLS = 0
RANDOM_WALLS = 1

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

def unroll_grid(state):
    # TODO fix the reshape once we switch to CNN
    # this is just a temp placeholder
    if torch.is_tensor(state):
        state = state.numpy()
    w, h = state.shape
    
    s = state.reshape((1, w*h )) + \
        np.random.rand(1, w*h)/10.0 
    s = torch.from_numpy(s).float() 
    return s

def main():
    NUM_TIMESTEPS = 10000
    MAX_MOVES_PER_GAME = 100
    AGENT_TYPE = DQN_AGENT
    PLAYER_START = RANDOM_START
    WALLS = STATIC_WALLS

    # create the game object
    game = GridWorld(10, 10, random_board=WALLS, random_start=PLAYER_START, max_moves_per_game=MAX_MOVES_PER_GAME)

    # create the agent
    visualize_game = False
    if AGENT_TYPE == RANDOM_AGENT:
        agent = Agent(action_space=game.action_space)

    elif AGENT_TYPE == DQN_AGENT:
        model = Model(init_grid_model(game.num_states, game.action_space))
        model.format_state = unroll_grid
        model.print()

        ### used to estimate the V map 
        # The vmap_estimation function expects a 10x10 grid without walls, where the player starts in the bottom 
        # left corner. This is because the function explictly steps through the gridworld with actions that traverse
        # every state (basically snaking up and down columns until it reaches the win). However, without a specific
        # obstacle free arrangement at the beginning of play, the game will end without hittign every state

        #model.load('model.pt')
        #grid_vmap_estimation = GridWorld(10,10, random_board=False,random_start=False, num_walls=0, static_start_pos = Position(0,9), max_moves_per_game=1000)
        #model.estimate_value_map(grid_vmap_estimation, save=True)

        agent = DQNAgent(model=model, action_space=game.action_space, training=True, batch_size=8, name='dqn')

    elif AGENT_TYPE == HUMAN_AGENT:
        visualize_game = True
        agent = HumanAgent("test", game.action_space)

    elif AGENT_TYPE == IQ_LEARN_AGENT:
        model = Model(init_grid_model(game.num_states, game.action_space))
        model.format_state = unroll_grid
        model.print()

        grid_vmap_estimation = GridWorld(10,10, random_board=False,random_start=False, num_walls=0, static_start_pos = Position(0,9), max_moves_per_game=1000)
        model.estimate_value_map(grid_vmap_estimation, save=True, path="bad_iqlearn_")

        expert_buffer = Buffer()
        expert_buffer.load_trajectories("good_dqn_2000.pkl", num_trajectories=50)
        agent = IQLearnAgent(model=model, action_space=game.action_space, training=True, epsilon=1, epsilon_floor=0.1)
        agent.format_state = unroll_grid
        agent.set_expert_buffer(expert_buffer)


    # create the orchestrator, which controls the game, with the game and agent objects
    orchestrator = Orchestrator(game=game, agent=agent, num_timesteps=NUM_TIMESTEPS, visualize=visualize_game)
    
    # play num_games games with the game and agent objects
    orchestrator.play()

    # save the trajectories of play from the games
    orchestrator.save_trajectories(filepath=f"{agent.name}_{NUM_TIMESTEPS}.pkl")

    # plot distance ratios
    orchestrator.plot_distance_ratios(save=True)

    if agent.has_model():
        # plot the model's losses
        agent.model.plot_losses(save=True)

    grid_vmap_estimation = GridWorld(10,10, random_board=False,random_start=False, num_walls=0, static_start_pos = Position(0,9), max_moves_per_game=1000)
    model.estimate_value_map(grid_vmap_estimation, save=True, path="bad_iqlearn_")

if __name__ == "__main__":
    main()
