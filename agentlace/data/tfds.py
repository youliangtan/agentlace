from __future__ import annotations

import gym
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Optional

from agentlace.data.data_store import DataStoreBase

##############################################################################


def make_datastore(
    dataset_dir, capacity: int, type="replay_buffer"
) -> DataStoreBase:
    """
    Load an RLDS dataset from the specified directory and populate it
    into the given datastore.

    Args:
        - dataset_dir: Directory where the RLDS dataset is stored.
        - capacity: Capacity of the replay buffer.
        - type: supported types are "replay_buffer" and "trajectory_buffer"

    Returns:
        - datastore: Datastore populated with the RLDS dataset.
    """
    # Load the dataset
    dataset = tfds.builder_from_directory(dataset_dir).as_dataset(split='all')

    # get the obs and action spec from the dataset
    obs_tensor_spec = dataset.element_spec["steps"].element_spec["observation"]
    action_tensor_spec = dataset.element_spec['steps'].element_spec['action']

    print("obs spec: ", obs_tensor_spec)
    print("action spec: ", action_tensor_spec)

    if type == "replay_buffer":
        from agentlace.data.jaxrl_data_store import ReplayBufferDataStore
        datastore = ReplayBufferDataStore(
            observation_space=tensor_spec_to_gym_space(obs_tensor_spec),
            action_space=tensor_spec_to_gym_space(action_tensor_spec),
            capacity=capacity,
        )
    elif type == "trajectory_buffer":
        from agentlace.data.jaxrl_data_store import make_default_trajectory_buffer
        datastore = make_default_trajectory_buffer(
            observation_space=tensor_spec_to_gym_space(obs_tensor_spec),
            action_space=tensor_spec_to_gym_space(action_tensor_spec),
            capacity=capacity,
        )
    else:
        raise ValueError(f"Unsupported type: {type}")

    return populate_datastore(datastore, dataset, type)


def populate_datastore(
    datastore: DataStoreBase,
    dataset: tf.data.Dataset,
    type: Optional[str] = None
) -> DataStoreBase:
    """
    Populate the given datastore with the RLDS dataset

    Args:
        - datastore: Replay buffer to populate.
        - dataset: RLDS dataset.
        - type: optional, additional support for 'trajectory_buffer' and w'ith_dones'
    Returns:
        - datastore: Datastore populated with the RLDS dataset.
    """
    # Iterate over episodes in the dataset
    for episode in dataset:
        steps = episode['steps']
        obs = None
        # Iterate through steps in the episode
        for i, step in enumerate(steps):
            if i == 0:
                obs = get_numpy_from_tensor(step['observation'])
                continue

            # Extract relevant data from the step
            next_obs = get_numpy_from_tensor(step['observation'])
            action = get_numpy_from_tensor(step['action'])
            reward = step.get('reward', 0).numpy()     # Defaulting to 0 if 'reward' key is missing
            terminate = step['is_terminal'].numpy()  # or is_last
            truncate = step['is_last'].numpy()  # truncate is not avail in the ReplayBuffer

            data = dict(
                observations=obs,
                next_observations=next_obs,
                actions=action,
                rewards=reward,
                masks=1 - terminate,  # 1 is transition, 0 is terminal
            )

            if type == "trajectory_buffer":
                data["end_of_trajectory"] = terminate or truncate
            elif type == "with_dones":
                data["dones"] = terminate or truncate

            # Insert data into the replay buffer
            datastore.insert(data)
            obs = next_obs
    return datastore

##############################################################################


def tensor_spec_to_gym_space(tensor_spec: tf.data.experimental.TensorSpec):
    """
    Convert a TensorSpec to a gym.Space, should support dict and box
    """
    if isinstance(tensor_spec, tf.TensorSpec):
        return gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=tensor_spec.shape,
            dtype=tensor_spec.dtype.as_numpy_dtype,
        )
    elif isinstance(tensor_spec, dict):
        return gym.spaces.Dict(
            {
                k: tensor_spec_to_gym_space(v)
                for k, v in tensor_spec.items()
            }
        )
    else:
        raise TypeError(f"Unsupported tensor spec type: {type(tensor_spec)}")


def get_numpy_from_tensor(tensor):
    """
    Convert a tensor to numpy
    """
    if isinstance(tensor, dict):
        return {k: get_numpy_from_tensor(v) for k, v in tensor.items()}
    return tensor.numpy()
