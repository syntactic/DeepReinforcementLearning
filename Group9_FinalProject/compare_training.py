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

def unroll_grid(state):
    if torch.is_tensor(state):
        state = state.numpy()
    w, h = state.shape
    
    s = state.reshape((1, w*h )) + \
        np.random.rand(1, w*h)/10.0 
    s = torch.from_numpy(s).float() 
    return s

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

def train_agent(agent_type, num_timesteps, batch_size):
    """ train an agent over num_timesteps with a batch size of batch_size"""

    # create the gridworld game
    grid = GridWorld(width=10, height=10)

    # make the model
    model = Model(init_grid_model(grid.num_states, grid.action_space))
    model.format_state = unroll_grid

    # create the agent based on the agent_type
    if agent_type == "DQN":
        agent = DQNAgent(model=model, action_space=grid.action_space, 
                            training=True, batch_size=batch_size, name='dqn_agent', 
                            epsilon=1, epsilon_floor=0.1)
    elif agent_type == "IQLearn":
        expert_buffer = Buffer()
        expert_buffer.load_trajectories("good_dqn_2000.pkl")
        agent = IQLearnAgent(model=model, action_space=grid.action_space, 
                            training=True, batch_size=batch_size, name='IQLearn_agent', 
                            epsilon=1, epsilon_floor=0.1)
        agent.set_expert_buffer(expert_buffer)
        print(agent.name)
    else:
        raise NotImplementedError(f"agent_type {agent_type} is not implemented. Only accepts 'DQN' or 'IQLearn'")
    
    # create an orchestrator to train the dqn_agent, which controls the game, with the game and agent objects
    orchestrator = Orchestrator(game=grid, agent=agent, num_timesteps=num_timesteps, visualize=False)
    
    # play num_games games with the game and agent objects
    orchestrator.play()

    # save the trajectories of play from the games
    orchestrator.save_trajectories(filepath=f"compare_training_results/{agent.name}_{num_timesteps}_{batch_size}.pkl")

    # get distance ratios for in the training of the dqn_orchestrator
    dist_rats = orchestrator.distance_ratios

    # get model loss
    loss = model.loss_bucket

    return dist_rats, loss

def compare_training(num_timesteps, batch_size, save=True):
    """ train agents and return losses and distance ratios """

    dqn_dist_rats, dqn_loss = train_agent("DQN", num_timesteps, batch_size)
    IQ_dist_rats, IQ_loss = train_agent("IQLearn", num_timesteps, batch_size)
    
    losses = {'DQN':dqn_loss, 'IQLearn':IQ_loss}
    dist_rats = {'DQN':dqn_dist_rats, 'IQLearn':IQ_dist_rats}

    if save:
        save_training_results(losses, dist_rats, num_timesteps, batch_size)

    return losses, dist_rats

def save_training_results(losses, dist_rats, num_timesteps, batch_size, folder_path="compare_training_results/"):
    """ save training results to pickle """
    losses_filepath = folder_path + f"losses_{num_timesteps}_{batch_size}.pkl"
    with open(losses_filepath, 'wb') as file:
        pickle.dump(losses, file)
    
    dist_rats_filepath = folder_path + f"distance_ratios_{num_timesteps}_{batch_size}.pkl"
    with open(dist_rats_filepath, 'wb') as file:
        pickle.dump(dist_rats, file)

def plot_losses(losses, name="", folder_path="compare_training_results/", save=True):
    """ plot the losses for each model by timestep """
    # set up the figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    # plot losses for each model
    for k in losses.keys():
        ax.plot(range(len(losses[k])), np.array(losses[k])/np.max(losses[k]))
        ax.set_xlabel('training steps')
        ax.set_ylabel('normalized loss')
    
    # create a figure legent
    ax.legend(losses.keys())

    # save the plot to an .svg
    if save:
        plt.tight_layout()
        fig.savefig(folder_path + "losses" + name + '.svg', format='svg', dpi=1200, bbox_inches='tight')
    
    # show the plot
    plt.show()

def plot_distance_ratios(dist_rats, name="", folder_path="compare_training_results/", save=True):
    """ plot the distance ratios for each model by game """
    # set up the figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    # plot losses for each model
    for k in dist_rats.keys():
        ax.plot(range(len(dist_rats[k])), dist_rats[k])
        ax.set_xlabel('games')
        ax.set_ylabel('distance ratio')
    
    # create a figure legent
    ax.legend(dist_rats.keys())

    # save the plot to an .svg
    if save:
        plt.tight_layout()
        fig.savefig(folder_path + "distance_ratios" + name + '.svg', format='svg', dpi=1200, bbox_inches='tight')
    
    # show the plot
    plt.show()

def main():

    """ NOTE for Tim, it seems like even if I play the same number of training steps
        with the same batch size, IQLearn model has less items in loss. How curious."""
    num_timesteps = 20000
    batch_size = 8

    losses, dist_rats = compare_training(num_timesteps, batch_size)

    plot_losses(losses)
    plot_distance_ratios(dist_rats)



if __name__ == "__main__":
    main()