#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, asdict

from typing import Dict, Optional, Tuple, List

from edgeml.data.data_store import DataStoreBase
from edgeml.data.sampler import make_jit_insert, make_jit_sample, Sampler


DATA_PREFIX = "data/"
METADATA_PREFIX = "metadata/"


@dataclass
class DataShape:
    name: str
    shape: Tuple[int, ...] = ()
    dtype: str = "float32"


class ReplayBuffer(DataStoreBase):
    """
    An in-memory data store for storing and sampling data
    from a (typically trajectory) dataset.
    # TODO (YL): support native numpy method.
    """

    def __init__(
        self,
        capacity: int,
        data_shapes: List[DataShape],
        device: Optional[jax.Device] = None,
        seed: int = 0,
        min_trajectory_length: int = 2,
    ):
        """
        Args:
            capacity: the maximum number of data points that can be stored
            data_shapes: a list of DataShape objects describing the data to be stored
            device: the device to store the data on. If None, defaults to the first CPU device.
            seed: the random seed to use for sampling
            min_trajectory_length: the minimum length of a trajectory to store
        """

        if device is None:
            device = jax.devices("cpu")[0]

        # Do it on CPU
        with jax.default_device(device):
            self.dataset = {
                _ds.name: jnp.zeros((capacity, *_ds.shape), dtype=_ds.dtype)
                for _ds in data_shapes
            }
            self._sample_rng = jax.random.PRNGKey(seed=seed)

        self.metadata = {
            "ep_begin": np.zeros((capacity,), dtype=jnp.int32),
            "ep_end": np.full((capacity,), -1, dtype=jnp.int32),
            "trajectory_id": np.full((capacity,), -1, dtype=jnp.int32),
            "seq_id": np.full((capacity,), -1, dtype=jnp.int32),
        }

        self.capacity = capacity
        self.size = 0
        self._latest_seq_id = 0
        self._trajectory = Trajectory(begin_idx=0, id=0, min_length=min_trajectory_length)
        self._sample_begin_idx = 0
        self._sample_end_idx = 0
        self._insert_idx = 0
        self._device = device

        self._insert_impl = make_jit_insert(device)
        self._sample_impls = {}

    def register_sample_config(
        self,
        name: str,
        samplers: Dict[str, Sampler],
        sample_range: Tuple[int, int] = (0, 1)
    ):
        assert (
            sample_range[1] - sample_range[0] > 0
        ), f"Sample range {sample_range} must be positive"
        assert (
            sample_range[1] - sample_range[0] <= self._trajectory.min_length
        ), f"Sample range {sample_range} must be <= the minimum trajectory length {self._trajectory.min_length}"
        self._sample_impls[name] = make_jit_sample(samplers, self._device, sample_range)

    def insert(self, data: Dict[str, jax.Array], end_of_trajectory: bool):
        """
        Insert a single data point into the data store.
        """
        self._latest_seq_id += 1  # TODO overflow issue?

        # Grab the metadata of the sample we're overwriting
        real_insert_idx = self._insert_idx % self.capacity
        overwritten_ep_end = self.metadata["ep_end"][real_insert_idx]

        with jax.default_device(self._device):
            self.dataset = self._insert_impl(
                self.dataset, data, real_insert_idx)
        # for k in self.dataset.keys():  # equivalent to _insert_impl above
        #     self.dataset[k] = self.dataset[k].at[real_insert_idx].set(data[k])

        self.metadata["ep_begin"][real_insert_idx] = self._trajectory.begin_idx
        self.metadata["ep_end"][real_insert_idx] = -1
        self.metadata["trajectory_id"][real_insert_idx] = self._trajectory.id
        self.metadata["seq_id"][real_insert_idx] = self._latest_seq_id

        self._insert_idx += 1
        self._sample_begin_idx = max(
            self._sample_begin_idx, self._insert_idx - self.capacity
        )

        # If there's not enough remaining of the overwritten trajectory to be valid, mark it as invalid
        if (
            overwritten_ep_end != -1
            and overwritten_ep_end - self._sample_begin_idx
            < self._trajectory.min_length
        ):
            self._sample_begin_idx = overwritten_ep_end

        # We don't want to sample from this trajectory until there's enough data to be valid
        if self._trajectory.valid(self._insert_idx):
            self._sample_end_idx = self._insert_idx

        self.size = min(self.size + 1, self.capacity)

        if end_of_trajectory:
            self.end_trajectory()

    def end_trajectory(self):
        """
        End a trajectory without inserting any data.
        """
        if not self._trajectory.valid(self._insert_idx):
            # If necessary, roll back the insert index to the beginning of the current trajectory if it's too short
            # We never set ep_end for this trajectory, so no need to rewrite it
            self._insert_idx = self._trajectory.begin_idx
        else:
            # This trajectory is long enough. Mark it as valid.
            self.metadata["ep_end"][
                self._trajectory.begin_idx: self._insert_idx
            ] = self._insert_idx

            # Update the metadata for the next trajectory
            self._trajectory.begin_idx = self._insert_idx
            self._trajectory.id += 1

    def sample(
        self,
        sampler_name: str,
        batch_size: int,
        force_indices: Optional[jax.Array] = None,
    ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
        """
        Sample a batch of data from the data store.

        Args:
            sampler_name: the name of the sampler
            batch_size: the batch size
            force_indices: if not None, force the sample to use these indices instead of sampling randomly.
                Assumed to be of shape (batch_size,) and smaller than the last valid index in the data store.
                
        Return:
            sampled_data: a dict of str-array pairs
            mask: a dict of str-array pairs indicating which data points are valid
        """
        if sampler_name not in self._sample_impls:
            raise ValueError(f"Sampler {sampler_name} not registered")

        rng, key = jax.random.split(self._sample_rng)
        sample_impl = self._sample_impls[sampler_name]
        sampled_data, mask = sample_impl(
            dataset=self.dataset,
            metadata=self.metadata,
            rng=key,
            batch_size=batch_size,
            sample_begin_idx=self._sample_begin_idx,
            sample_end_idx=self._sample_end_idx,
            sampled_idcs=force_indices,
        )
        self._sample_rng = rng
        return sampled_data, mask

    def serialized(self):
        dataset_dict = {
            f"{DATA_PREFIX}{k}": np.asarray(v[: self.size]) for k, v in self.dataset.items()
        }
        metadata_dict = {
            f"{METADATA_PREFIX}{k}": np.asarray(v[: self.size])
            for k, v in self.metadata.items()
        }
        # TODO: _sample_begin_idx, _sample_end_idx are not serialized?
        return {
            **dataset_dict,
            **metadata_dict,
            **self._trajectory.to_dict(),
            "size": self.size,
            "capacity": self.capacity,
            "_insert_idx": self._insert_idx,
        }

    def save(self, path: str):
        """Save a data store to a file."""
        np.savez(path, **self.serialized())

    @classmethod
    def load(
        self,
        path: str,
        device: Optional[jax.Device] = None,
    ):
        """Load a data store from a file. Returns a ReplayBuffer object."""
        loaded_data = np.load(path)
        return self.deserialize(loaded_data, device)

    @staticmethod
    def deserialize(
        loaded_data: Dict,
        device: Optional[jax.Device] = None,
    ):
        """Load a stream of data from a file. Returns a ReplayBuffer object."""
        capacity = loaded_data["capacity"]
        size = loaded_data["size"]
        insert_idx = loaded_data["_insert_idx"]
        data = {
            k.split("/")[1]: v
            for k, v in loaded_data.items()
            if k.startswith(DATA_PREFIX)
        }
        metadata = {
            k.split("/")[1]: v
            for k, v in loaded_data.items()
            if k.startswith(METADATA_PREFIX)
        }

        data_shapes = [
            DataShape(name=k, shape=v.shape[1:], dtype=str(v.dtype)) for k, v in data.items()
        ]
        replay_buffer = ReplayBuffer(capacity, data_shapes, device=device)
        replay_buffer._trajectory = Trajectory.from_dict(loaded_data)
        replay_buffer._latest_seq_id = np.max(metadata["seq_id"])
        replay_buffer.size = size
        replay_buffer._insert_idx = insert_idx

        replay_buffer.dataset = jax.tree_map(
            lambda dataset, data: dataset.at[:size].set(data), replay_buffer.dataset, data
        )
        for k, v in metadata.items():
            replay_buffer.metadata[k][:size] = v
        return replay_buffer

    def latest_data_id(self) -> int:
        """return the lastest data id, which is the seq id"""
        return self._latest_seq_id

    def get_latest_data(self, from_id: int
                        ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """
        Note that this data is a form of compact data,
            :return indices, data in dict of str-array pairs
        """
        # Find all indices where the seq_d is greater than the provided seq_id
        indices = jnp.where(self.metadata["seq_id"] > from_id)[0]

        # Extract data for those indices
        dataset_dict = {
            f"{DATA_PREFIX}{k}": jnp.asarray(v[indices]) for k, v in self.dataset.items()
        }
        metadata_dict = {
            f"{METADATA_PREFIX}{k}": jnp.asarray(v[indices]) for k, v in self.metadata.items()
        }
        # NOTE: this will resulted in the Trainer's datastore being readonly since
        #       local stateful variables e.g. traj are not provided in this method
        data = {
            **dataset_dict,
            **metadata_dict
        }
        return indices, data

    def update_data(self, indices: jax.Array, data: Dict[str, jax.Array]):
        """
        This method partially update data of the ReplayBuffer,
        in accordance to the indices provided in the data
        """
        # TODO (YL): improve this loop operation
        # Update dataset with the provided data
        for k, v in self.dataset.items():
            updated_values = data[f"{DATA_PREFIX}{k}"]
            assert len(updated_values) == len(indices)
            # Use JAX operations to update values at the specified indices
            with jax.default_device(self._device):
                self.dataset[k] = v.at[indices].set(updated_values)

        # Update metadata with the provided metadata
        for k, v in self.metadata.items():
            updated_values = data[f"{METADATA_PREFIX}{k}"]
            assert len(updated_values) == len(indices)
            v[indices] = updated_values

        # get largest seq_id in the metadata["seq_id"]
        self._latest_seq_id = np.max(self.metadata["seq_id"])

    def __len__(self):
        """Get the number of valid data points in the data store."""
        return len(np.where(self.metadata["seq_id"] > 0)[0])

################################################################################


@dataclass
class Trajectory:
    """This is used internally for tracking the current trajectory."""
    begin_idx: int
    id: int
    min_length: int

    def valid(self, curent_idx: int) -> bool:
        return curent_idx - self.begin_idx >= self.min_length

    def to_dict(self) -> Dict[str, int]:
        return {f"traj/{k}": v for k, v in asdict(self).items()}

    @staticmethod
    def from_dict(data: Dict[str, int]):
        return Trajectory(
            begin_idx=data["traj/begin_idx"],
            id=data["traj/id"],
            min_length=data["traj/min_length"]
        )
