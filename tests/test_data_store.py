#!/usr/bin/env python3

import pytest
import jax
import jax.numpy as jnp
from agentlace.data.trajectory_buffer import DataShape
from agentlace.data.trajectory_buffer import TrajectoryBuffer
from agentlace.data.sampler import LatestSampler, SequenceSampler
import random


@pytest.fixture
def data_store(capacity=100):
    ds = TrajectoryBuffer(
        capacity=capacity,
        data_shapes=[
            DataShape(name="data"),
            DataShape(name="index", dtype="int32"),
            DataShape(name="trajectory_id", dtype="int32"),
        ],
    )

    ds.register_sample_config(
        "default",
        {
            "data": LatestSampler(),
            "index": LatestSampler(),
            "trajectory_id": LatestSampler(),
        },
    )

    return ds


def helper_insert_trajectory(
    data_store: TrajectoryBuffer, trajectory_length: int, traj_id: int, end_trajectory=True
):
    for i in range(trajectory_length):
        data = {
            "data": random.random(),
            "index": i,
            "trajectory_id": traj_id,
        }
        data_store.insert(data, False)

    if end_trajectory:
        data_store.end_trajectory()


def helper_check_ep_metadata(data_store: TrajectoryBuffer):
    dataset_size = data_store.capacity

    for i in range(data_store._sample_begin_idx, data_store._sample_end_idx):
        data, metadata = jax.tree_map(
            lambda x: x[i % dataset_size], (data_store.dataset, data_store.metadata)
        )

        ep_begin = metadata["ep_begin"]
        ep_end = metadata["ep_end"]

        ep_begin = max(ep_begin, data_store._sample_begin_idx)

        if ep_end == -1:
            ep_end = data_store._sample_end_idx

        assert ep_begin <= i < ep_end, "Index should be within episode range"

        if i + 1 != ep_end:
            assert (
                data_store.metadata["trajectory_id"][i % dataset_size]
                == data_store.metadata["trajectory_id"][(i + 1) % dataset_size]
            ), "Trajectory ID should be the same in an episode"
        else:
            assert (
                data_store.metadata["trajectory_id"][i % dataset_size]
                != data_store.metadata["trajectory_id"][(i + 1) % dataset_size]
            ), "Trajectory ID should be different across episodes"


def test_data_store_basic_insert_retrieve(data_store: TrajectoryBuffer):
    helper_insert_trajectory(data_store, 10, 0)
    helper_insert_trajectory(data_store, 15, 1)
    helper_insert_trajectory(data_store, 20, 2)

    helper_check_ep_metadata(data_store)

    samples, samples_valid = data_store.sample(
        "default", 64, force_indices=jnp.arange(45)
    )
    assert samples["data"].shape[0] == 45


def test_data_store_insert_wrap(data_store: TrajectoryBuffer):
    helper_insert_trajectory(data_store, 10, 0)
    helper_insert_trajectory(data_store, 15, 1)
    helper_insert_trajectory(data_store, 20, 2)
    helper_insert_trajectory(data_store, 75, 3)

    helper_check_ep_metadata(data_store)


def test_data_store_insert_sequence(data_store: TrajectoryBuffer):
    data_store.register_sample_config(
        "sequence",
        {
            "index_prev": SequenceSampler(begin=-3, end=1, source="index"),
            "index_future": SequenceSampler(begin=0, end=4, source="index"),
            "index": LatestSampler(),
            "trajectory_id": LatestSampler(),
        },
    )

    helper_insert_trajectory(data_store, 10, 0)
    helper_insert_trajectory(data_store, 15, 1)
    helper_insert_trajectory(data_store, 20, 2)
    helper_insert_trajectory(data_store, 75, 3)

    data, valid = data_store.sample(
        "sequence",
        100,
        force_indices=jnp.arange(
            data_store._sample_begin_idx, data_store._sample_end_idx
        ),
    )

    # Trajectory 0 should be 100% overwritten
    assert jnp.all(data["trajectory_id"] != 0)

    # The oldest data should have invalid previous trajectories
    first_trajectory_begin = (data["trajectory_id"] == 1)[:, None] & (
        (data["index"][:, None] + jnp.arange(-3, 1)) < 10
    )
    assert not jnp.any(valid["index_prev"] & first_trajectory_begin)

    # The newest data should have invalid future trajectories
    last_trajectory_end = (data["trajectory_id"] == 2)[:, None] & (
        (data["index"][:, None] + jnp.arange(0, 4)) >= 75
    )
    assert not jnp.any(valid["index_future"] & last_trajectory_end)


def test_sample_trajectory_short_valid(data_store: TrajectoryBuffer):
    data_store._trajectory.min_length = 3

    data_store.register_sample_config(
        "need_padding",
        {
            "index": LatestSampler(),
        },
        (-1, 2),
    )

    helper_insert_trajectory(data_store, 40, 0)
    helper_insert_trajectory(data_store, 40, 1)
    helper_insert_trajectory(data_store, 40, 2)

    data, valid = data_store.sample(
        "need_padding",
        100,
        force_indices=jnp.arange(
            data_store._sample_begin_idx, data_store._sample_end_idx
        ),
    )

    assert jnp.all(valid["index"] > 0)
    assert jnp.all(valid["index"] < 40)


def test_sample_trajectory_too_short(data_store: TrajectoryBuffer):
    data_store._trajectory.min_length = 3
    data_store.register_sample_config(
        "need_padding",
        {
            "index": LatestSampler(),
            "trajectory_id": LatestSampler(),
        },
        (-1, 2),
    )

    helper_insert_trajectory(data_store, 10, 0)
    helper_insert_trajectory(data_store, 2, 1)

    data, valid = data_store.sample(
        "need_padding",
        100,
    )

    assert jnp.all(data["trajectory_id"] == 0)


def test_trajectory_becomes_too_short_from_overwrite(data_store: TrajectoryBuffer):
    data_store._trajectory.min_length = 3

    data_store.register_sample_config(
        "need_padding",
        {
            "index": LatestSampler(),
            "trajectory_id": LatestSampler(),
        },
        (-1, 2),
    )

    for i in range(10):
        helper_insert_trajectory(data_store, 10, i)
    helper_insert_trajectory(data_store, 8, 10)

    assert data_store._sample_begin_idx == 10
    assert data_store._sample_end_idx == 108

    data, valid = data_store.sample(
        "need_padding",
        100,
        force_indices=jnp.arange(
            data_store._sample_begin_idx, data_store._sample_end_idx
        ),
    )

    assert jnp.all(data["trajectory_id"] != 0)


def test_in_progress_trajectory(data_store: TrajectoryBuffer):
    data_store._trajectory.min_length = 3

    data_store.register_sample_config(
        "need_padding",
        {
            "index": LatestSampler(),
            "trajectory_id": LatestSampler(),
        },
        (-1, 2),
    )

    helper_insert_trajectory(data_store, 10, 0)
    helper_insert_trajectory(data_store, 10, 1, end_trajectory=False)

    data, valid = data_store.sample(
        "need_padding",
        100,
    )

    assert jnp.any(data["trajectory_id"] == 1)


def test_in_progress_trajectory_too_short(data_store: TrajectoryBuffer):
    data_store._trajectory.min_length = 3

    data_store.register_sample_config(
        "need_padding",
        {
            "index": LatestSampler(),
            "trajectory_id": LatestSampler(),
        },
        (-1, 2),
    )

    helper_insert_trajectory(data_store, 10, 0)
    helper_insert_trajectory(data_store, 1, 1, end_trajectory=False)

    data, valid = data_store.sample(
        "need_padding",
        100,
        force_indices=jnp.arange(
            data_store._sample_begin_idx, data_store._sample_end_idx
        ),
    )

    assert jnp.all(data["trajectory_id"] == 0)

def test_save_and_load_file(data_store: TrajectoryBuffer):
    for i in range(10):
        helper_insert_trajectory(data_store, 5, i)
    for i in range(8):
        helper_insert_trajectory(data_store, 8, i)
        
    assert data_store._sample_begin_idx == 15
    assert data_store._sample_end_idx == 5*10 + 8*8
    
    data_store.save("test.npz")
    
    data_store2 = TrajectoryBuffer.load("test.npz")
    # TODO: fix this? _sample_begin_idx and end is not saved
    # assert data_store2._sample_begin_idx == data_store._sample_begin_idx
    assert data_store2._insert_idx == data_store._insert_idx
    assert data_store2.capacity == data_store.capacity

    # check arbitrary data
    assert jnp.all(data_store2.dataset["data"] == data_store.dataset["data"])
