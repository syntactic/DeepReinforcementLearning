import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from util_classes import *

# tile types
FLOOR = 0
WALL = 1
AGENT = 2
WIN = 3

# possible actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorld:

    def __init__(self, agent, width, height, seed=0):
        self.agent = agent
        self.width = width
        self.height = height
        self.seed = seed
        self.win_state = Position(width-1, height-1)
        self.agent.pos = Position(0,0)
        self.num_walls = 5
        self.state = self.generate_grid()
    

    def generate_grid(self) -> np.ndarray:
        """Generate the grid environment."""

        # initialize an empty grid with the correct dimensions
        grid = np.zeros((self.width, self.height))

        # place agent 
        grid[self.agent.pos.x, self.agent.pos.y] = AGENT

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
    
    def check_possible(self, action):
        """checks if an action is possible given the current state"""
        possible = False

        # get current position of agent
        curr_pos = self.agent.pos

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
        if(self.check_possible(action)):
            self.state[self.agent.pos.y, self.agent.pos.x] = FLOOR
            self.agent.pos = self.update_position(self.agent.pos, action)
            self.state[self.agent.pos.y, self.agent.pos.x] = AGENT


        # still need to return all the stuff needed for deep learning like reward, transition, game_won?, etc


    
    def visualize_grid(self):
        """plots the current state of the grid"""

        # discrete color map to interpret the grid: floor, wall, agent, win
        cmap = colors.ListedColormap(['#CACACA', '#625151', '#E53A3A', '#FFE400'])
        
        # plot the grid as an image
        plt.imshow(self.state, cmap=cmap)
        plt.show()

        

         