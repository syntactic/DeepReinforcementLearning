import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from utils import *

# tile types
FLOOR = 0
WALL = 1
PLAYER = 2
WIN = 3

# possible actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorld:

    def __init__(self, width=10, height=10, seed=0, max_moves_per_game=100):
        self.width = width
        self.height = height
        self.seed = seed
        self.win_state = Position(width-1, height-1)
        self.player_pos = Position(0,0)
        self.num_walls = 5
        self.state = self.generate_grid()
        self.action_space = np.array([UP, RIGHT, DOWN, LEFT])
        self.num_states = self.width * self.height
        self.max_moves_per_game = max_moves_per_game
        self.moves_made = 0
    
    def reset(self, seed = 0, player_pos = Position(0,0)):
        """ 
        resets the state of the grid and moves the agent to start position 
        (agent_pos) 
        """
        self.player_pos = player_pos
        self.seed = seed
        self.state = self.generate_grid()
        self.moves_made = 0


    def generate_grid(self) -> np.ndarray:
        """Generate the grid environment."""

        # initialize an empty grid with the correct dimensions
        grid = np.zeros((self.width, self.height))

        # place agent 
        grid[self.player_pos.x, self.player_pos.y] = PLAYER

        # place win tile 
        grid[self.win_state.x, self.win_state.y] = WIN

        # randomly place walls
        np.random.seed(self.seed)
        walls_placed = 0
        while walls_placed < self.num_walls:
            x=np.random.choice(grid.shape[0], 1)
            y=np.random.choice(grid.shape[1], 1)

            if(grid[x,y] == FLOOR):
                grid[x,y] = WALL
                walls_placed+=1

        return grid
    
    def check_game_over(self):
        """ checks if the player is in the win state or if the max moves were taken"""
        return (self.player_pos.x == self.win_state.x and \
            self.player_pos.y == self.win_state.y) or \
            self.moves_made >= self.max_moves_per_game
    
    def reward(self):
        """ calculates the reward based on the current state of the gridworld """
        if (self.player_pos.x == self.win_state.x and \
            self.player_pos.y == self.win_state.y):
            return 10
        else:
            return -1
    
    def check_possible(self, action):
        """checks if an action is possible given the current state"""
        possible = False

        # get current position of the player
        curr_pos = self.player_pos

        # get new position resulting from action
        new_pos = self.update_position(curr_pos, action)

        # check if new_pos is within bounds and unoccupied
        within_bounds = (new_pos.x >=0 and new_pos.x < self.width) and \
                        (new_pos.y >=0 and new_pos.y < self.height)
        if(within_bounds):
           possible = not(self.state[new_pos.y, new_pos.x] == WALL)
        
        return possible
        
    def update_position(self, pos, action):
        """ applys the given action to a position pos"""

        if action == UP:
            new_pos = Position(pos.x, pos.y-1)
        if action == RIGHT:
            new_pos = Position(pos.x+1, pos.y)
        if action == DOWN:
            new_pos = Position(pos.x, pos.y+1)
        if action == LEFT:
            new_pos = Position(pos.x-1, pos.y)

        return new_pos

    def step(self, action):
        """Transition the current state into the next state given an action"""
        self.moves_made +=1

        if(self.check_possible(action)):
            self.state[self.player_pos.y, self.player_pos.x] = FLOOR
            self.player_pos = self.update_position(self.player_pos, action)
            self.state[self.player_pos.y, self.player_pos.x] = PLAYER

        return (self.state, self.reward(), self.check_game_over())
    
    def get_state(self):
        """ return state of the game"""
        return self.state
    
    def get_player_position(self):
        """ returns the player position """
        return self.player_pos
    
    def set_player_position(self, player_pos):
        """ sets the player position """
        self.player_pos = player_pos

    def visualize_grid(self):
        """plots the current state of the grid"""

        # discrete color map to interpret the grid: floor, wall, agent, win
        cmap = colors.ListedColormap(['#CACACA', '#625151', '#E53A3A', '#FFE400'])
        
        # plot the grid as an image
        plt.imshow(self.state, cmap=cmap)
        plt.show()

    def grid_image(self):
        """ generates an image of the grid (grid.width, grid.height, 3) (3 channels for rbg) """

        grid_image = np.array([self.state, self.state, self.state]).T

        colors = [[202,202,202], [98, 81, 81], [229, 58, 58], [225, 228, 0]]
        
        for i in range(len(colors)):
            grid_image[grid_image[:,:,0] == i] = colors[i]

        return grid_image
        

         
