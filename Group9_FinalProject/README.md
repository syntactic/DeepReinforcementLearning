# Deep Reinforcement Learning Group 9 Final Project
This project takes the core algorithmic idea of [IQ-Learn](https://div99.github.io/IQ-Learn) from Garg et al.'s paper [IQ-Learn: Inverse soft-Q Learning for Imitation](https://arxiv.org/abs/2106.12142) and implements it a Python-based GridWorld environment.

Authors: [Celine Aubuchon](https://github.com/celine-aubuchon) and [Timothy Ho](https://github.com/syntactic) 

[Project Report](https://github.com/syntactic/DeepReinforcementLearning/blob/main/Group9_FinalProject/ImplementingInverseQLearning_Aubuchon_Ho.pdf)

[Video Presentation](https://github.com/syntactic/DeepReinforcementLearning/blob/main/Group9_FinalProject/ImplementingInverseQLearning_Aubuchon_Ho.m4v)

## Features
 - GridWorld with customizable levels of difficulty:
    * width and height parameters
    * number of wall obstacles
    * randomization of player placement
    * randomization of wall placement
    * randomization of win state placement
 - four kinds of agents:
    1. random agent (serves as a reasonable baseline)
    2. human agent (for understanding and debugging the environment and for collecting expert data)
    3. DQN agent
    4. IQ-Learn agent
        - IQ loss function for offline learning, no gradient penalty or divergence functions
- trajectory saving 
- GUI for interactive playing and model visualization
- plotting and analysis scripts

## Requirements
- numpy
- torch
- matplotlib
- pygame
## Usage

The `main.py` script can be run and by default it will train a DQN online over the course of 5000 timesteps. Additional arguments can be supplied to change some parameters at runtime:
 - `-t NUM_TIMESTEPS` the total number of timesteps that the Orchestrator will make the player play (default 5000)
 - `-m MAX_MOVES` the maximum number of timesteps allowed per game (default 100)
 - `--height HEIGHT` the number of rows in the GridWorld
 - `--width WIDTH` the number of columns in the GridWorld
 - `-rs` whether to have the player spawn in a random location each game
 - `-rw` whether to have the walls spawn in random locations each game
 - `-rws` whether to have the win state spawn in a random location each game
 - `agent_type {random, human, dqn, iq}` the kind of agent that will play. If iq is selected, additional parameters must be supplied:
    - `-d PATH_TO_DEMONSTRATION_PICKLE` path to a Pickle file containing a dictionary of states, actions, rewards, next states, and dones
    - `-nt NUM_TRAJECTORIES` the number of trajectories from the Pickle to load into the expert buffer. The default size of the buffer is 2048 (timesteps), so if your trajectories average 10 timesteps each, you could fit around 200 trajectories into the buffer. Choosing less will limit the amount of data that is used for training.

Example usage:

`python main.py -rw -rs -t 20000 iq -d expert_data/human_5000_random_walls_random_win_state.pkl -nt 500`

This would train an IQ-Learn agent on data containing 500 trajectories with random walls and random win states. As it's training, the model will be providing actions to a GridWorld of default dimensions (10x10) that have random wall locations and random start state locations. This will continue for 20,000 timesteps. At the end of training, the script will automatically plot:
- distance ratios per game
- loss over training steps
- value map over the GridWorld
- reward map (using the inferred reward from IQ-Learn's loss function) over the GridWorld

