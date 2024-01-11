#!/usr/bin/env python3

import random
from agentlace.data.trajectory_buffer import DataShape
from agentlace.data.jaxrl_data_store import TrajectoryBufferDataStore

from agentlace.data.sampler import LatestSampler
from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer, TrainerConfig
import argparse
import time
import logging


print_green = lambda x: print("\033[92m {}\033[00m" .format(x))

def create_trajectory_buffer():
    # Step 1. Create the TrajectoryBuffer object
    buffer_capacity = 100
    data_shapes = [
        DataShape(name="data"),
        DataShape(name="index", dtype="int32"),
        DataShape(name="trajectory_id", dtype="int32"),
    ]
    replay_buffer = TrajectoryBufferDataStore(capacity=buffer_capacity, data_shapes=data_shapes)

    # Step 2. Register the sampler config
    sample_config = {
        "data": LatestSampler(),
        "index": LatestSampler(),
        "trajectory_id": LatestSampler(),
    }
    replay_buffer.register_sample_config("default", sample_config)

    return replay_buffer


def insert_data(data_store):
    """Step 3. Insert data into the replay buffer."""""
    for traj_id in range(3):  # Inserting 3 trajectories
        for idx in range(10):  # Each trajectory has 10 data points
            data = {
                "data": random.random(),
                "index": idx,
                "trajectory_id": traj_id,
            }
            data_store.insert(data)

        # TODO: end of trajectory should be handle internally in TrajectoryBuffer
        # data_store.end_trajectory()


def sample_data(replay_buffer):
    """Step 4. Sample data from the replay buffer."""
    samples, samples_valid = replay_buffer.sample("default", 5)

    # pretty print the sampled data
    print(f"replay buffer has {len(replay_buffer)} data points")
    print("Sampled data as {key: value}:")
    for key in samples.keys():
        print_green(f" - {key}: {samples[key]}")


##############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--client', action='store_true')
    parser.add_argument('--ip', type=str, default='localhost')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    replay_buffer = create_trajectory_buffer()
    t_config = TrainerConfig(port_number=50051, broadcast_port=50052)

    if args.server:
        print("Starting server, use replay buffer")
        server = TrainerServer(t_config)
        server.register_data_store("table1", replay_buffer)
        server.start(threaded=True)
        print("Server started")
        try:
            while True:
                sample_data(replay_buffer)
                time.sleep(5)
        except KeyboardInterrupt:
            print("Stopping client")
            server.stop()

    elif args.client:
        print("starting client, use queue data store")
        # we can mix and matche datastore on trainer server and client
        queue_data_store = QueuedDataStore(10)
        client = TrainerClient("table1", args.ip, t_config, queue_data_store)
        client.start_async_update(3)
        # keyboard interrput
        try:
            while True:
                insert_data(queue_data_store)
                time.sleep(3)
        except KeyboardInterrupt:
            print("Stopping client")
            client.stop()

    else:
        insert_data(replay_buffer)
        sample_data(replay_buffer)

    print("Done")
