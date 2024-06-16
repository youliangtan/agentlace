#!/usr/bin/env python3

from agentlace.data.tfds import make_datastore, populate_datastore
from agentlace.data.jaxrl_data_store import ReplayBufferDataStore
from agentlace.data.jaxrl_data_store import make_default_trajectory_buffer

from oxe_envlogger.rlds_logger import RLDSLogger

import gym
from gym import spaces
import numpy as np
import os

class CustomEnv(gym.Env):
    """
    A custom environment that uses a dictionary for the observation space.
    """

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            'velocity': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })

    def step(self, action):
        observation = {
            'position': np.random.randint(0, 10, size=(1,)),
            'velocity': np.random.uniform(-1, 1, size=(1,))
        }
        # Example reward, done, and info
        reward = 1.0
        done = False
        info = {}
        return observation, reward, done, info

    def reset(self):
        # Example initial observation
        observation = {
            'position': np.random.randint(0, 10, size=(1,)),
            'velocity': np.random.uniform(-1, 1, size=(1,))
        }
        return observation, {}


def run_rlds_logger(env,
                    capacity=20,
                    log_dir = "logs/test_rlds_env",
                    type="replay_buffer"):

    logger = RLDSLogger(
        observation_space=env.observation_space,
        action_space=env.action_space,
        dataset_name="test_rlds_env",
        directory=log_dir,
        max_episodes_per_file=5,  # TODO: arbitrary number
    )

    if type == "replay_buffer":
        data_store = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=capacity,
            rlds_logger=logger,
        )
    elif type == "trajectory_buffer":
        data_store = make_default_trajectory_buffer(
            env.observation_space,
            env.action_space,
            capacity=capacity,
            rlds_logger=logger,
        )

    # create some fake data
    sample_obs = env.reset()[0]
    action_shape = env.action_space.shape

    TOTAL_EPISODES = 10
    STEPS_PER_EPISODE = 15

    print("inserting data")
    for j in range(TOTAL_EPISODES):
        for i in range(STEPS_PER_EPISODE):
            done = 0 if i < 14 else 1 # last sample is terminal
    
            sample = dict(
                observations=sample_obs,
                next_observations=sample_obs,
                actions=np.random.randn(*action_shape),
                rewards=np.random.randn(),
                masks=1 - done, # 1 is transition, 0 is terminal
            )

            if type == "trajectory_buffer":
                sample["end_of_trajectory"] = False

            data_store.insert(sample)
    logger.close()

    # check if log dir has more than 3 files
    files = os.listdir(log_dir)
    assert len(files) == 4, "expected 2 tfrecord files, and 2 json config files"

    # check if the data is stored correctly
    total_idx_sum = 0
    def data_transform(data, metadata):
        nonlocal total_idx_sum
        total_idx_sum += metadata["step"]
        return data

    # This will create a new datastore with the same log_dir to do a sanity check
    stored_buffer = make_datastore(
        log_dir,
        capacity=200,
        type=type,
        data_transform=data_transform,
    )
    print("total data size: ", len(stored_buffer))
    expected_total_idx_sum = list(range(STEPS_PER_EPISODE))*TOTAL_EPISODES
    assert sum(expected_total_idx_sum) == total_idx_sum
    assert len(stored_buffer) == STEPS_PER_EPISODE * TOTAL_EPISODES


if __name__ == "__main__":
    env = CustomEnv()
    run_rlds_logger(env)

    print("testing pendulum env")
    env = gym.make("Pendulum-v1")
    run_rlds_logger(env)
    
    # NOTE: trajectory buffer only support obs and action space
    # of type array (not dict)
    env = gym.make("HalfCheetah-v4")
    run_rlds_logger(env, type="trajectory_buffer")
    print("all tests passed")
