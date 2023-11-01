import pytest
from DQNAgent import *
from GridWorld import *
from main import init_grid_model, unroll_grid
from constants import *

@pytest.fixture
def ten_by_ten_static_grid():
    game = GridWorld(10, 10, random_board=STATIC_WALLS, random_start=STATIC_START, max_moves_per_game=100)
    return game

@pytest.fixture()
def dqn_agent(ten_by_ten_static_grid):
    model = Model(init_grid_model(ten_by_ten_static_grid.num_states, ten_by_ten_static_grid.action_space))
    model.format_state = unroll_grid
    agent = DQNAgent(model=model, action_space=ten_by_ten_static_grid.action_space, training=False, batch_size=8, name='good_dqn', epsilon=0, epsilon_floor=0)
    return agent

def test_action(ten_by_ten_static_grid, dqn_agent):
    action = dqn_agent.get_action(ten_by_ten_static_grid.get_state())
    assert action >= 0 and action <= ten_by_ten_static_grid.action_space.size

def test_has_model(ten_by_ten_static_grid, dqn_agent):
    assert dqn_agent.has_model()
