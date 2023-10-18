import numpy as np
import matplotlib.pyplot as plt

from DQN import DQN
from GridWorld import GridWorld
from Agent import Agent

import torch



### NOTES 
"""
- there seems to be some difference in the way I am calcuting average reward across 
    game for training versus testing -> should check this out and fix it
- right now, the whole design is not right because the agent class does nothing, 
    it is all controlled through DQN and GridWorld
- Tim suggests using some kind of orchestrator that controls the gridworld, and the Agent (which could be a vanilla dqn,
    a random agent, IQ learn agent, expert, etc)
- check if it makes sense to use maxQ in reward calculation
- change to accumulate losses over each game and take an average
- consider CNN because it may capture position information rather than fully-connected layers

"""

def main():
    NUM_EPOCHS = 1
    GAMMA = 0.9
    MAX_MOVES_PER_GAME = 5
    dqn = DQN(GridWorld(10, 10), 
              num_epochs=NUM_EPOCHS, 
              gamma=GAMMA, 
              max_moves_per_game=MAX_MOVES_PER_GAME)

    print(dqn.model)

    dqn.train()

    dqn.plot_training_results(save=True, name='training1')

    seeds, rewards, results = dqn.test(num_games=100, max_moves=100)

    dqn.plot_test_results(seeds, rewards, results)




if __name__ == "__main__":
    main()