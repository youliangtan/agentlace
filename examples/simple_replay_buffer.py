#!/usr/bin/env python3

import random
from edgeml.data.replay_buffer import ReplayBuffer, DataShape
from edgeml.data.sampler import LatestSampler


def main():
    # 1. Create the ReplayBuffer object
    buffer_capacity = 100
    data_shapes = [
        DataShape(name="data"),
        DataShape(name="index", dtype="int32"),
        DataShape(name="trajectory_id", dtype="int32"),
    ]
    replay_buffer = ReplayBuffer(capacity=buffer_capacity, data_shapes=data_shapes)

    # 2. Register the sampler config
    sample_config = {
        "data": LatestSampler(),
        "index": LatestSampler(),
        "trajectory_id": LatestSampler(),
    }
    replay_buffer.register_sample_config("default", sample_config)

    # 3. Insert data
    for traj_id in range(3):  # Inserting 3 trajectories
        for idx in range(10):  # Each trajectory has 10 data points
            data = {
                "data": random.random(),
                "index": idx,
                "trajectory_id": traj_id,
            }
            replay_buffer.insert(data, False)
        replay_buffer.end_trajectory()

    # 4. Sample 5 data points
    samples, samples_valid = replay_buffer.sample("default", 5)

    # pretty print the sampled data
    print("Sampled data as {key: value}")
    for key in samples.keys():
        print(f" - {key}: {samples[key]}")


if __name__ == "__main__":
    main()
