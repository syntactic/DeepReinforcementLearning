import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib import colors
from utils import *
from constants import *


class GridWorld:

    def __init__(self, width=10, height=10, random_board=False, random_start=False, random_win_state=False, seed=0, num_walls=5, max_moves_per_game=100, static_start_pos=Position(0,0)):
        self.width = width
        self.height = height
        self.random_start = random_start
        self.random_board = random_board
        self.random_win_state = random_win_state
        self.static_start_pos = static_start_pos
        self.seed = seed
        self.win_state = Position(width-1, height-1)
        self.player_pos = self.new_player_pos()
        self.num_walls = num_walls
        self.state = self.generate_grid()
        self.action_space = np.array([UP, RIGHT, DOWN, LEFT])
        self.num_states = self.width * self.height
        self.max_moves_per_game = max_moves_per_game
        self.moves_made = 0

    @classmethod
    def from_state(cls, state, win_state):
        """
        This takes a state (2d numpy array) and creates an underspecified
        GridWorld from it. It's meant to be used for the reward recovery.
        """
        shape = state.shape

        g = cls()
        g.height = shape[0]
        g.width = shape[1]
        g.player_pos = Position(*index_of_value_in_2d_array(state, PLAYER)[::-1]) # ugly way of getting col, row
        g.win_state = win_state
        g.state = state

        return g

    
    def new_player_pos(self):
        if self.random_start:
            while True:
                np.random.seed()
                x = np.random.choice(self.width, 1)[0]
                y = np.random.choice(self.height, 1)[0]
                if x != self.win_state.x or y != self.win_state.y:
                    return Position(x,y)
        else:
            return self.static_start_pos


    def reset(self):
        """ 
        resets the state of the grid and moves the agent to start position 
        (agent_pos) 
        """
        self.player_pos = self.new_player_pos()
        if self.random_board:
            self.seed += 1
        self.state = self.generate_grid()
        while self.is_impossible():
            self.seed += 1
            self.state = self.generate_grid()
        self.moves_made = 0

    def is_impossible(self):
        win_next_to_left_wall = self.win_state.x == 0
        win_next_to_right_wall = self.win_state.x == self.width-1
        win_next_to_top_wall = self.win_state.y == 0
        win_next_to_bottom_wall = self.win_state.y == self.height-1

        if not win_next_to_left_wall:
            if self.state[self.win_state.y, self.win_state.x-1] in [FLOOR, PLAYER]:
                return False
        if not win_next_to_right_wall:
            if self.state[self.win_state.y, self.win_state.x+1] in [FLOOR, PLAYER]:
                return False
        if not win_next_to_top_wall:
            if self.state[self.win_state.y-1, self.win_state.x] in [FLOOR, PLAYER]:
                return False
        if not win_next_to_bottom_wall:
            if self.state[self.win_state.y+1, self.win_state.x] in [FLOOR, PLAYER]:
                return False
        has_player = index_of_value_in_2d_array(self.state, PLAYER) != -1
        has_win = index_of_value_in_2d_array(self.state, WIN) != -1
        return has_player and has_win

    def generate_random_position(self):
        x=np.random.choice(self.width, 1)[0]
        y=np.random.choice(self.height, 1)[0]
        return Position(x, y)

    def generate_grid(self) -> np.ndarray:
        """Generate the grid environment."""

        # initialize an empty grid with the correct dimensions
        grid = np.zeros((self.width, self.height))

        # place agent 
        grid[self.player_pos.y, self.player_pos.x] = PLAYER

        # place win tile
        if self.random_win_state:
            self.win_state = self.generate_random_position()
        grid[self.win_state.y, self.win_state.x] = WIN

        def check_win_blocked(grid, new_wall_x, new_wall_y):
            """ 
            checks if the win is still accessible with the placement of the new
            wall

            NOTE wondering whether this is necessary, because it doesn't check
            if the agent has a path to the way anyway, only if the win is immediately
            surrounded
            """
            win_x = self.win_state.x; win_y = self.win_state.y 

            # pad the grid with wall tiles
            g = np.pad(grid,(1,1), 'constant', constant_values=(WALL,WALL)) 

            # add the new wall to the grid 
            g[new_wall_y, new_wall_x] = WALL

            # get sum of the tiles including and surrounding the win_state
            sum_of_tiles = np.sum(np.sum(g[win_y-1:win_y+2, win_x-1:win_x+2]))

            # return whether walls would completely surround the win by checking
            # that the sum of WALL tiles + WIN is >= that case
            return sum_of_tiles >= 8*WALL + WIN


        # randomly place walls
        np.random.seed(self.seed)
        walls_placed = 0
        while walls_placed < self.num_walls:
            x=np.random.choice(grid.shape[0], 1)[0]
            y=np.random.choice(grid.shape[1], 1)[0]
            rand_pos = self.generate_random_position()

            if grid[rand_pos.y, rand_pos.x] == FLOOR:
                grid[rand_pos.y, rand_pos.x] = WALL
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
           possible = self.state[new_pos.y, new_pos.x] != WALL
        
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
        tile_below_player = FLOOR
        if self.player_pos == self.win_state:
            tile_below_player = WIN
        if(self.check_possible(action)):
            self.state[self.player_pos.y, self.player_pos.x] = tile_below_player
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
        
    def distance_from_agent_to_win_state(self):
        return abs(self.win_state.x - self.player_pos.x) + abs(self.win_state.y - self.player_pos.y)

    def get_full_state_space(self):
        """ returns all states """
        grid = np.copy(self.state)
        grid[self.player_pos.y, self.player_pos.x] = FLOOR

        states = []
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x] != WALL:
                    states.append(np.copy(grid))
                    states[-1][y, x] = PLAYER

        return states
