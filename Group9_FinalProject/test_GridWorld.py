import pytest
from GridWorld import *
from constants import *

@pytest.fixture
def four_by_four_static_grid():
    game = GridWorld(4, 4, random_board=STATIC_WALLS, random_start=STATIC_START, num_walls=0, max_moves_per_game=100)
    return game

@pytest.fixture
def four_by_four_static_grid_one_wall():
    game = GridWorld(4, 4, random_board=STATIC_WALLS, random_start=STATIC_START, num_walls=1, max_moves_per_game=100)
    return game

def test_move(four_by_four_static_grid):
    assert four_by_four_static_grid.state[0][0] == PLAYER
    four_by_four_static_grid.step(RIGHT)
    assert four_by_four_static_grid.state[0][1] == PLAYER

def test_reward(four_by_four_static_grid):
    four_by_four_static_grid.step(RIGHT)
    assert four_by_four_static_grid.reward() == -1

    four_by_four_static_grid.step(RIGHT)
    assert four_by_four_static_grid.reward() == -1

    four_by_four_static_grid.step(RIGHT)
    assert four_by_four_static_grid.reward() == -1

    four_by_four_static_grid.step(DOWN)
    assert four_by_four_static_grid.reward() == -1

    four_by_four_static_grid.step(DOWN)
    assert four_by_four_static_grid.reward() == -1

    four_by_four_static_grid.step(DOWN)
    assert four_by_four_static_grid.reward() == 10

def test_get_full_state_space(four_by_four_static_grid):
    state_space = four_by_four_static_grid.get_full_state_space()
    assert len(state_space) == 16
    assert state_space[0][0, 0] == PLAYER
    assert state_space[-1][3, 3] == PLAYER

def test_get_full_state_space_one_wall(four_by_four_static_grid_one_wall):
    state_space = four_by_four_static_grid_one_wall.get_full_state_space()
    assert len(state_space) == 15
    assert state_space[0][0, 0] == PLAYER
    assert state_space[-1][3, 3] == PLAYER
