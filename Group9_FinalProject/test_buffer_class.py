import numpy as np
import pytest
from utils import *

def test_buffer_size():
    # test size after adding samples
    b = Buffer()
    timestep = (np.zeros((10,10)), 0, -1, np.zeros((10,10)), False)

    assert b.size() == 0
    b.add(timestep)
    assert b.size() == 1
    b.add(timestep)
    assert b.size() == 2

def test_buffer_load_trajectories():
    # test size after loading samples
    b = Buffer()
    trajs_path = "testing_data/good_dqn_2000.pkl"

    ### the correct number of trajectories are loaded
    # load 25 trajectories and count the total trajectories
    b.load_trajectories(trajs_path, num_trajectories=25)
    dones = [b.data[i][4] for i in range(b.size())]
    counter = np.ones((len(dones), 1))
    traj_count = np.sum(counter[dones])

    assert traj_count == 25

    # add 5 more trajectories and count the total
    b.load_trajectories(trajs_path, num_trajectories=5)
    dones = [b.data[i][4] for i in range(b.size())]
    counter = np.ones((len(dones), 1))
    traj_count = np.sum(counter[dones])

    assert traj_count == 30



