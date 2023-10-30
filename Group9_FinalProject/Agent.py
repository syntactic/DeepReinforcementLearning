import numpy as np

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
    """
    def __init__(self, action_space:np.ndarray, name:str="rand"):
        self.name = name
        self.action_space = action_space
        self.state = None
        self.next_state = None
        self.reward = None
        self.game_over = None
        self.action = None
        self.training = False
    
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

    def reset(self):
        self.state = float("nan")
        self.next_state = float("nan")
        self.reward = float("nan")
        self.game_over = float("nan")

