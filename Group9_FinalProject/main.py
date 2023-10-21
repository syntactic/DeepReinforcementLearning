import numpy as np
import matplotlib.pyplot as plt

from DQN import DQN
from GridWorld import GridWorld
from Agent import Agent
from DQNAgent import DQNAgent
from HumanAgent import HumanAgent
from Orchestrator import Orchestrator

import torch

RANDOM_AGENT = 0
DQN_AGENT = 1
HUMAN_AGENT = 2

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

def main():
    NUM_GAMES = 1000
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
        model = init_grid_model(game.num_states, game.action_space)
        agent = DQNAgent(model=model, action_space=game.action_space)
    elif AGENT_TYPE == HUMAN_AGENT:
        visualize_game = True
        agent = HumanAgent("test", game.action_space)

    # create the orchestrator, which controls the game, with the game and agent objects
    orchestrator = Orchestrator(game=game, agent=agent, num_games=NUM_GAMES, visualize=visualize_game)
    
    # play num_games games with the game and agent objects
    orchestrator.play()

    # save the trajectories of play from the games
    orchestrator.save_trajectories()

    agent.print_results()


if __name__ == "__main__":
    main()
