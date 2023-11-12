import numpy as np
import matplotlib.pyplot as plt
import argparse

from GridWorld import GridWorld
from Agent import Agent
from DQNAgent import DQNAgent
from IQLearnAgent import IQLearnAgent
from HumanAgent import HumanAgent
from Orchestrator import Orchestrator
from utils import *
from Model import Model
from pathlib import Path

import torch
from constants import WALL, PLAYER, WIN

RANDOM_AGENT = 0
DQN_AGENT = 1
HUMAN_AGENT = 2
IQ_LEARN_AGENT = 3

STATIC_START = 0
RANDOM_START = 1

STATIC_WALLS = 0
RANDOM_WALLS = 1

DEFAULT_TIMESTEPS = 5000
DEFAULT_MAX_MOVES_PER_GAME = 100
DEFAULT_NUM_TRAJECTORIES = 100
DEFAULT_HEIGHT = 10
DEFAULT_WIDTH = 10

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

def create_argument_parser():
    parser = argparse.ArgumentParser(prog='GridWorld Player',
            description='Plays gridworld with any of several agent types: \
            human, random, DQN, IQ Learn.')
    parser.add_argument('-t', '--timesteps', default=DEFAULT_TIMESTEPS, type=int)
    parser.add_argument('-m', '--max_moves', default=DEFAULT_MAX_MOVES_PER_GAME, type=int)
    parser.add_argument('-rs', '--random_start', action='store_true', default=True)
    parser.add_argument('-rw', '--random_walls', action='store_true', default=False)
    parser.add_argument('-rws', '--random_win_state', action='store_true', default=False)
    parser.add_argument('--height', default=DEFAULT_HEIGHT, type=int)
    parser.add_argument('--width', default=DEFAULT_WIDTH, type=int)
    subparser = parser.add_subparsers(dest='agent_type')
    # this doesn't trigger any of the DQN subparser, so keep in mind if you want to add options for DQN
    subparser.default = 'dqn' 
    subparser.add_parser('human')
    subparser.add_parser('random')
    dqn_subparser = subparser.add_parser('dqn')
    iq_learn_subparser = subparser.add_parser('iq')
    iq_learn_subparser.add_argument('-d', '--data', required=True)
    iq_learn_subparser.add_argument('-nt', '--num_trajectories', default=DEFAULT_NUM_TRAJECTORIES, type=int)

    return parser

def main():
    AGENT_TYPE = IQ_LEARN_AGENT
    PLAYER_START = RANDOM_START
    WALLS = STATIC_WALLS

    parser = create_argument_parser()
    args = parser.parse_args()
    # may want to delete this once we know it's working correctly
    print(args)

    # create the game object
    game = GridWorld(args.width, args.height, random_board=args.random_walls, random_start=args.random_start,
            random_win_state=args.random_win_state, max_moves_per_game=args.max_moves)

    # create the agent
    visualize_game = False
    if args.agent_type == 'random':
        agent = Agent(action_space=game.action_space)

    elif args.agent_type == 'dqn':
        model = Model(init_grid_model(game.num_states, game.action_space))
        model.format_state = unroll_grid
        model.print()

        #model.load('model.pt')
        agent = DQNAgent(model=model, action_space=game.action_space, training=True, batch_size=8, name='dqn')

    elif args.agent_type == 'human':
        visualize_game = True
        agent = HumanAgent("test", game.action_space)

    elif args.agent_type == 'iq':
        model = Model(init_grid_model(game.num_states, game.action_space))
        model.format_state = unroll_grid
        model.print()

        expert_buffer = Buffer(memory_size=2048)
        expert_buffer.load_trajectories(args.data, num_trajectories=args.num_trajectories)
        agent = IQLearnAgent(model=model, action_space=game.action_space, training=True, epsilon=1, epsilon_floor=0.1)
        agent.format_state = unroll_grid
        agent.set_expert_buffer(expert_buffer)

    
    Path(f"./{agent.name}_{args.timesteps}").mkdir(parents=True, exist_ok=True)

    # create the orchestrator, which controls the game, with the game and agent objects
    orchestrator = Orchestrator(game=game, agent=agent, num_timesteps=args.timesteps, visualize=visualize_game)
    
    # play num_games games with the game and agent objects
    orchestrator.play()

    # save the trajectories of play from the games
    orchestrator.save_trajectories(filepath=f"{agent.name}_{args.timesteps}.pkl")

    # plot distance ratios
    orchestrator.plot_distance_ratios(save=True, path=f"{agent.name}_{args.timesteps}/")

    if agent.has_model():
        # plot the model's losses
        agent.model.plot_losses(save=True, path=f"{agent.name}_{args.timesteps}/")
        grid_vmap_estimation = GridWorld(args.width, args.height, random_board=False,random_start=False, num_walls=0, static_start_pos = Position(0,9), max_moves_per_game=1000)
        agent.model.estimate_reward_map(grid_vmap_estimation, save=True, path=f"{agent.name}_{args.timesteps}/trained_")
        agent.model.estimate_value_map(grid_vmap_estimation, save=True, path=f"{agent.name}_{args.timesteps}/trained_")

if __name__ == "__main__":
    main()
