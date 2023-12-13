# !/usr/bin/env python3
# NOTE: this requires jaxrl_m to be installed:
#       https://github.com/rail-berkeley/jaxrl_minimal

from threading import Lock
from typing import List, Optional
from edgeml.data.data_store import DataStoreBase
from edgeml.data.trajectory_buffer import TrajectoryBuffer, DataShape

import gym
import jax
from jaxrl_m.data.replay_buffer import ReplayBuffer


##############################################################################

class TrajectoryBufferDataStore(TrajectoryBuffer, DataStoreBase):
    def __init__(
        self,
        capacity: int,
        data_shapes: List[DataShape],
        device: Optional[jax.Device] = None,
        seed: int = 0,
        min_trajectory_length: int = 2,
    ):
        TrajectoryBuffer.__init__(
            self,
            capacity=capacity,
            data_shapes=data_shapes,
            min_trajectory_length=min_trajectory_length,
            device=device,
            seed=seed,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(TrajectoryBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(TrajectoryBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO
