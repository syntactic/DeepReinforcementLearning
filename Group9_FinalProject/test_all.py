import numpy as np
import pytest
from GridWorld import GridWorld

def test_gridworld_wallcount():
   wall_counts = [1, 3, 5, 10]
   for count in wall_counts:
      g = GridWorld(num_walls=count)
      assert count == len(np.where(g.state == 1))