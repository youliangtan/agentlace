from __future__ import annotations

import gym
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Optional, Callable

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
    type: Optional[str] = None,
    data_transform: Optional[Callable] = None,
) -> DataStoreBase:
    """
    Populate the given datastore with the RLDS dataset

    Args:
        - datastore: Replay buffer to populate.
        - dataset: RLDS dataset.
        - type: optional, additional support for 'trajectory_buffer' and 'with_dones'
        - data_transform: optional[callable], function to transform the data before
                        inserting into the datastore.
                        with format: data_transform(data: Dict, metadata: Dict) -> Dict

    Returns:
        - datastore: Datastore populated with the RLDS dataset.
    """
    # get the keys to used as metadata if needed by the data_transform() fn
    step_keys = dataset.element_spec["steps"].element_spec.keys()
    metadata_keys = [k for k in step_keys if k not in ['observation', 'action', 'reward', 'is_terminal', 'is_last']]
    # print("metadata_keys: ", metadata_keys)

    # Iterate over episodes in the dataset
    for episode in dataset:
        steps = episode['steps']
        obs = None
        _step_size = len(steps)
        # Iterate through steps in the episode
        for i, step in enumerate(steps):
            if i == 0:
                obs = get_value_from_tensor(step['observation'])
                continue

            # Extract relevant data from the step
            next_obs = get_value_from_tensor(step['observation'])
            action = get_value_from_tensor(step['action'])
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

            # Transform the data if user provided a data_transform function
            # NOTE: the arg data is a dict of the data inserted into the datastore
            # and the metadata is a dict of the metadata extracted from the step
            # which are not part of data
            if data_transform is not None:
                metadata = dict(
                    step=i,
                    step_size=_step_size,
                )
                for key in metadata_keys:
                    metadata[key] = get_value_from_tensor(step[key])
                data = data_transform(data, metadata)

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


def get_value_from_tensor(tensor):
    """
    Convert a tensor to numpy or other python types
    """
    if isinstance(tensor, dict):
        return {k: get_value_from_tensor(v) for k, v in tensor.items()}
    elif tensor.dtype == tf.string:
        return tensor.numpy().decode('utf-8')
    return tensor.numpy()
