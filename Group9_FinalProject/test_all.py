import numpy as np
import pytest
from utils import *
from GridWorld import GridWorld

def test_gridworld_wallcount():
   wall_counts = [1, 3, 5, 10]
   for count in wall_counts:
      g = GridWorld(num_walls=count)
      assert count == len(np.where(g.state.flatten() == 1)[0])

def test_gridworld_distance():
    g = GridWorld()
    # default distance
    assert g.distance_from_agent_to_win_state() == 18

    # set agent
    p = Position(5, 8)
    g.player_pos = p
    assert g.distance_from_agent_to_win_state() == 5
