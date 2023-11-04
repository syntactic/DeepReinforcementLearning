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

@pytest.fixture
def four_by_four_static_grid_player_above_win(four_by_four_static_grid):
    player_pos = four_by_four_static_grid.player_pos
    win_state = four_by_four_static_grid.win_state
    four_by_four_static_grid.state[player_pos.x, player_pos.y] = FLOOR
    player_pos = Position(win_state.x, win_state.y-1)
    four_by_four_static_grid.player_pos = player_pos
    return four_by_four_static_grid

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

def test_move_out_of_win_state(four_by_four_static_grid_player_above_win):
    # these modifications are necessary to set up the test
    state = four_by_four_static_grid_player_above_win.state
    win_state = four_by_four_static_grid_player_above_win.win_state
    player_pos = four_by_four_static_grid_player_above_win.player_pos
    assert player_pos.y == win_state.y - 1 and player_pos.x == win_state.x

    four_by_four_static_grid_player_above_win.step(DOWN)
    assert four_by_four_static_grid_player_above_win.player_pos == four_by_four_static_grid_player_above_win.win_state
    assert state[3, 3] == PLAYER

    four_by_four_static_grid_player_above_win.step(UP)
    assert player_pos.y == win_state.y - 1 and player_pos.x == win_state.x
    assert state[3, 3] == WIN
    assert state[2, 3] == PLAYER

def test_create_gridworld_from_state():
    state = np.array([[PLAYER, FLOOR], [FLOOR, WIN]])
    g = GridWorld.from_state(state, Position(1,1))
    assert np.array_equal(state, g.state)
