#!/usr/bin/env python3

import numpy as np

# check if jax is installed
try:
    import jax
    import jax.numpy as np
    import chex
except ImportError:
    print("JAX is not installed, revert back to numpy")


from typing import Dict, Tuple, Optional
from enum import IntEnum


##############################################################################


class Sampler:
    """Abstract base class for samplers."""
    source = None  # default source name

    def sample(self,
               sampled_idx: np.ndarray,
               ep_begin: np.ndarray,
               ep_end: np.ndarray,
               key: np.ndarray,
               dataset: Dict[str, np.ndarray],
               source_name: str
               ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Sample from the data according to the config.

        Returns:
            data: the sampled data of shape (batch_size, ...) if a single element is selected,
                or (batch_size, time, ...) if a sequence is selected (e.g. history)
            mask: a boolean mask of the sampled data with 
                shape (batch_size, ...) or (batch_size, time, ...)
        """
        raise NotImplementedError

    def _access(self, idx: np.ndarray, ep_begin, ep_end, dataset, source_name):
        # check if source is defined, and if the source name is in dataset
        # else use the default source name which is ownself
        # TODO: better and clearner implementation for indexing
        if self.source is not None:
            source_name = self.source
            if source_name not in dataset:
                raise ValueError("Bad sampling config: {config} has \
                    source {source_name} which is not in the dataset")

        dataset_size = dataset[source_name].shape[0]
        mask_ep_begin = expand_to_shape(ep_begin, idx.shape)
        mask_ep_end = expand_to_shape(ep_end, idx.shape)

        data = dataset[source_name][
            np.clip(idx, mask_ep_begin, mask_ep_end) % dataset_size
        ]
        return data, (idx >= mask_ep_begin) & (idx < mask_ep_end)

##############################################################################


class LatestSampler(Sampler):
    def sample(self, sampled_idx, ep_begin, ep_end, key, dataset, source_name):
        return self._access(sampled_idx, ep_begin, ep_end, dataset, source_name)


##############################################################################

class SequenceSampler(Sampler):
    def __init__(self, squeeze=False, begin=0, end=1, source=None,
                 device=jax.devices("cpu")[0]):
        self.squeeze = squeeze
        self.seq_begin = begin
        self.seq_end = end
        self.device = device
        self.source = source

    def sample(self, sampled_idx, ep_begin, ep_end, key, dataset, source_name):
        sequence_len = self.seq_end - self.seq_begin
        assert sequence_len > 0, f"History length must be positive, got {sequence_len}"
        with jax.default_device(self.device):
            indices = (
                np.arange(self.seq_begin, self.seq_end)[None, :] + sampled_idx[:, None]
            )
        batch_size = sampled_idx.shape[0]
        chex.assert_shape(indices, (batch_size, sequence_len))

        if self.squeeze:
            assert (
                sequence_len == 1
            ), f"Can only squeeze sequence if length is 1, but got {sequence_len}"
            indices = np.squeeze(indices, axis=-1)

        return self._access(indices, ep_begin, ep_end, dataset, source_name)


##############################################################################

class FutureSampler(Sampler):

    class Distribution(IntEnum):
        UNIFORM = 0
        EXPONENTIAL = 1

    def __init__(self, distribution=Distribution.UNIFORM, max_future=None,
                 lambda_=None, source=None):
        """Probability distribution for sampling future indices."""
        self.distribution = distribution
        self.max_future = max_future
        self.lambda_ = lambda_
        self.source = source

    def sample(self, sampled_idx, ep_begin, ep_end, key, dataset, source_name):
        if self.distribution == self.Distribution.UNIFORM:
            # Set max_future to sample from [t, t + max_future)
            max_future_length = self.max_future
            # If max_future is None, set it to the end of the episode.
            # Otherwise, clip to the end of the episode.
            if max_future_length is None:
                max_future = ep_end
            else:
                max_future = np.minimum(ep_end, sampled_idx + max_future_length)

            future_indices = jax.random.randint(
                key, shape=sampled_idx.shape, minval=sampled_idx, maxval=max_future)

        elif self.distribution == self.Distribution.EXPONENTIAL:
            offset = jax.random.exponential(key, shape=sampled_idx.shape) * self.lambda_
            future_indices = offset.astype(int) + sampled_idx
            future_indices = np.minimum(future_indices, ep_end - 1)
        else:
            raise ValueError(f"Unknown distribution")

        return self._access(future_indices, ep_begin, ep_end, dataset, source_name)

##############################################################################


class PrioritySampler(Sampler):
    def __init__(self, priorities: np.ndarray, alpha=1.0, source=None):
        """
        Initializes the PrioritySampler with given priorities.
        # TODO complete this

        Args:
        - priorities: An array of priority values where the index in the 
                      array corresponds to the index in the dataset.
                      Higher values mean higher priority.
        - alpha: Exponent to which the priorities are raised before normalization.
                 This can be used to skew the distribution of samples.
        - source: The source from which to sample.
        """
        self.priorities = priorities
        self.alpha = alpha
        self.source = source
        self._update_probs()
        raise NotImplementedError("PrioritySampler is not fully implemented yet")

    def _update_probs(self):
        """Updates the sampling probabilities based on the priorities."""
        probs = np.power(self.priorities, self.alpha)
        self.probs = probs / np.sum(probs)

    def sample(self, sampled_idx, ep_begin, ep_end, key, dataset, source_name):
        indices = jax.random.choice(key, a=len(self.probs), p=self.probs,
                                    shape=sampled_idx.shape)
        return self._access(indices, ep_begin, ep_end, dataset, source_name)

    def update_priorities(self, idx, priorities):
        """
        Update the priorities for the given indices.

        Args:
        - idx (list or np.ndarray): Indices to update.
        - priorities (list or np.ndarray): Corresponding new priorities.
        """
        self.priorities[idx] = priorities
        self._update_probs()


##############################################################################


def expand_to_shape(x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Expand an array with correct prefix dimensions to a given shape.
    """
    assert x.ndim <= len(shape)
    assert shape[: x.ndim] == x.shape, f"Bad shape {shape} for {x}"

    while x.ndim < len(shape):
        x = x[..., None].repeat(shape[x.ndim], axis=-1)

    return x


def make_jit_sample(sample_config: dict, device: jax.Device, sample_range: Tuple[int, int]):
    """
    Make a JIT-compiled sample function for a dataset, according to the config.
    """

    def _sample_impl(
        dataset: jax.Array,
        metadata: Dict[str, jax.Array],
        rng: jax.Array,
        batch_size: int,
        sample_begin_idx: int,
        sample_end_idx: int,
        sampled_idcs: Optional[jax.Array] = None,
    ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
        indices_key, *sampling_keys = jax.random.split(
            rng, len(sample_config.keys()) + 1
        )
        sampling_keys = {k: v for k, v in zip(sample_config.keys(), sampling_keys)}

        if sampled_idcs is None:
            sampled_idcs = jax.random.randint(
                indices_key,
                shape=(batch_size,),
                minval=sample_begin_idx,
                maxval=sample_end_idx,
                dtype=np.int32,
            )

        ep_begins = np.maximum(metadata["ep_begin"][sampled_idcs], sample_begin_idx)
        ep_ends = np.minimum(metadata["ep_end"][sampled_idcs], sample_end_idx)
        sampled_idcs = np.clip(
            sampled_idcs, ep_begins - sample_range[0], ep_ends - sample_range[1]
        )

        result = {
            k: sampler.sample(
                sampled_idx=sampled_idcs,
                source_name=k,
                dataset=dataset,
                ep_begin=ep_begins,
                ep_end=ep_ends,
                key=sampling_keys[k],
            )
            for k, sampler in sample_config.items()
        }

        sampled_data = {k: v[0] for k, v in result.items()}
        samples_valid = {k: v[1] for k, v in result.items()}

        return sampled_data, samples_valid

    return jax.jit(_sample_impl, static_argnames=("batch_size",), device=device)


def make_jit_insert(device: jax.Device):
    """
    Make a JIT-compiled insert function for a dataset.
    """

    def _insert_tree_impl(
        dataset: Dict[str, jax.Array], data: Dict[str, jax.Array], insert_idx: int
    ) -> Dict[str, jax.Array]:
        """
        Insert the new data into the dataset at the specified index, return the new dataset.
        """
        # Check should never run after JIT
        # chex.assert_trees_all_equal_shapes_and_dtypes(
        #     data, jax.tree_map(lambda x: x[0], dataset)
        # )
        return jax.tree_map(
            lambda k_dataset, k_data: k_dataset.at[insert_idx].set(k_data),
            dataset, data
        )

    return jax.jit(_insert_tree_impl, donate_argnums=(0,), device=device)
