import numpy as np
import matplotlib.pyplot as plt

from DQN import DQN
from GridWorld import GridWorld
from Agent import Agent
from DQNAgent import DQNAgent
from Orchestrator import Orchestrator

import torch

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
    game = GridWorld(10, 10, max_moves_per_game=MAX_MOVES_PER_GAME)
    model = init_grid_model(game.num_states, game.action_space) # change this to use get methods
    agent = DQNAgent(model=model, action_space=game.action_space)
    #agent = Agent(action_space=game.action_space)

    orchestrator = Orchestrator(game=game, agent=agent, num_games=NUM_GAMES)
    
    orchestrator.play()

    agent.print_results()


if __name__ == "__main__":
    main()