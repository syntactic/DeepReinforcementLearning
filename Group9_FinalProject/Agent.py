import numpy as np
import matplotlib.pyplot as plt

class Agent:
    """
    A class to represent an agent.

    The default Agent class takes random actions based on
    the action_space attribute and does not learn based on
    experience.

    ...

    Attributes
    ----------
    action_space : numpy array
        an array of total possible actions

    Methods
    -------
    get_action(state):
        Returns an action given a state of the game.
    
    inform_result(next_state, reward, game_won):
        Updates the agent's internal parameters based on an inputted state, 
        reward, and bool indicating whether the game was won. This assumes
        that the next_state, reward, and bool are a result of the agent's 
        most recent action

    reset():
        resets the Agent's internal parameters to reflect the beginning of 
        a new game. Note, this does NOT reset any learning from the previous
        game, if applicable.
    
    print_results():
        plots current results of the agent (depends on agent what type of results)

    save_results():
        This saves the data relevant to the agent. For ex,
        for a human agent, it saves the trajectories of the games, while for a 
        DQN agent, it will save the trained model
    """
    def __init__(self, action_space:np.ndarray):
        self.action_space = action_space
        self.state = float("nan")
        self.next_state = float("nan")
        self.reward = float("nan")
        self.game_over = float("nan")
        self.action = float("nan")
    
    def get_action(self, state):

        # TODO: decide if a random seed should be set explictly here, so
        # that this random sampling does not interfere/ interact with seed 
        # setting studd elsewhere
        self.state = state
        self.action = np.random.choice(self.action_space, size=1)[0]
        return self.action

    def inform_result(self, next_state, reward, game_over):

        self.next_state = next_state
        self.reward = reward
        self.game_over = game_over

        pass
    
    def reset(self):
        self.state = float("nan")
        self.next_state = float("nan")
        self.reward = float("nan")
        self.game_over = float("nan")
    
    def print_results(self):
        pass
    def save_results(self):
        """ saves the trajectory of the agent """
        pass
