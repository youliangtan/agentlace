"""
Requirements:
    dlimp:  git clone git@github.com:kvablack/dlimp.git
    tensorflow_datasets: pip install tensorflow-datasets
    tf-agents-nightly: pip install tf-agents
"""

from functools import partial
from typing import Any, Optional
from agentlace.data.rlds_writer import RLDSWriter
import tensorflow as tf
from tf_agents.replay_buffers.episodic_replay_buffer import (
    EpisodicReplayBuffer,
    StatefulEpisodicReplayBuffer,
)
from agentlace.data.data_store import DataStoreBase
from dlimp.dataset import DLataset, _wrap as dlimp_wrap

class EpisodicTFDataStore(DataStoreBase):
    def __init__(
        self,
        capacity: int,
        data_spec: tf.TypeSpec,
        rlds_logger: Optional[RLDSWriter] = None,
    ):
        super().__init__(capacity)

        def _begin_episode(trajectory):
            return trajectory["is_first"][..., 0]

        def _end_episode(trajectory):
            return trajectory["is_last"][..., -1]

        self._num_data_seen = 0
        data_spec = {
            **data_spec,
            "_traj_index": tf.TensorSpec(shape=(), dtype=tf.int64),
        }
        self._replay_buffer = StatefulEpisodicReplayBuffer(
            EpisodicReplayBuffer(
                data_spec=data_spec,
                capacity=capacity,
                buffer_size=1,
                begin_episode_fn=_begin_episode,
                end_episode_fn=_end_episode,
            )
        )

        self._logger = rlds_logger

    def insert(self, data: Any):
        self._replay_buffer.add_sequence(
            tf.nest.map_structure(
                partial(tf.expand_dims, axis=0),
                data | {"_traj_index": self._replay_buffer.episode_ids},
            )
        )
        self._num_data_seen += 1

        if self._logger:
            self._logger(data)

    @property
    def size(self):
        return min(self._num_data_seen, self.capacity)

    @partial(dlimp_wrap, is_flattened=False)
    def as_dataset(self) -> DLataset:
        dataset: tf.data.Dataset = self._replay_buffer.as_dataset()

        def convert_trajectory_to_dlimp(trajectory, aux):
            ep_len = tf.shape(tf.nest.flatten(trajectory)[0])[0]
            return {
                **trajectory,
                "_len": tf.repeat(ep_len, ep_len),
                "_frame_index": tf.range(ep_len),
            }

        return dataset.map(convert_trajectory_to_dlimp)


if __name__ == "__main__":
    data_spec = {
        "action": tf.TensorSpec(shape=(2,), dtype=tf.float32),
        "observation": {
            "image": tf.TensorSpec(shape=(), dtype=tf.string),
            "position": tf.TensorSpec(shape=(2,), dtype=tf.float32),
        },
        "is_first": tf.TensorSpec(shape=(), dtype=tf.bool),
        "is_last": tf.TensorSpec(shape=(), dtype=tf.bool),
        "is_terminal": tf.TensorSpec(shape=(), dtype=tf.bool),
        "reward": tf.TensorSpec(shape=(), dtype=tf.float32),
        "discount": tf.TensorSpec(shape=(), dtype=tf.float32),
    }

    DATA_DIRECTORY = "/tmp/test_rlds_agentlace"
    
    import os
    import shutil

    # Delete the directory if it exists and create a new one
    if os.path.exists(DATA_DIRECTORY):
        shutil.rmtree(DATA_DIRECTORY)
    os.makedirs(DATA_DIRECTORY)

    logger = RLDSWriter(
        "test_rlds_agentlace",
        data_spec,
        data_directory=DATA_DIRECTORY,
        version="0.0.1",
    )
    buffer = EpisodicTFDataStore(1000, data_spec, rlds_logger=logger)
    dataset = buffer.as_dataset()
    dataset_iter = dataset.flatten().batch(256).as_numpy_iterator()
    
    print("Inserting data")

    for episode in range(100):
        ep_len = tf.random.uniform(shape=(), minval=1, maxval=10, dtype=tf.int32)
        for i in range(ep_len):
            buffer.insert(
                {
                    "action": tf.constant([0, 0], dtype=tf.float32),
                    "observation": {
                        "image": tf.constant("image", dtype=tf.string),
                        "position": tf.constant([i, 0], dtype=tf.float32),
                    },
                    "is_first": tf.constant(i == 0, dtype=tf.bool),
                    "is_last": tf.constant(i == ep_len - 1, dtype=tf.bool),
                    "is_terminal": tf.constant(False, dtype=tf.bool),
                    "reward": tf.constant(1, dtype=tf.float32),
                    "discount": tf.constant(1, dtype=tf.float32),
                }
            )

    from tqdm import trange

    for _ in range(100):
        next(dataset_iter)
    for _ in trange(1000, desc="Benchmarking sampling speed..."):
        next(dataset_iter)

    # important to close the logger so that the 
    logger.close()
    
    ##########################################################################
    # Simple test cases
    ##########################################################################

    # read the dataset from the disk
    import tensorflow_datasets  as tfds
    dataset = tfds.builder_from_directory(DATA_DIRECTORY).as_dataset(split="all")

    print("size of dataset", len(list(dataset)))

    assert len(list(dataset)) == 100, "There should be 100 episodes in the dataset"
    for episode in dataset.take(1):
        steps = episode["steps"]
        for i, step in enumerate(steps):
            if i == 0:
                assert step["is_first"].numpy() == True, "first step is is_first=True"
            else:
                assert step["is_first"].numpy() == False, "non-first step is is_first=False"

            assert len(step["observation"]) == 2, "observation should have 2 keys"

            print(" is first and last: ", step["is_first"].numpy(), step["is_last"].numpy())
            is_last = step["is_last"]
    print("Done All")
