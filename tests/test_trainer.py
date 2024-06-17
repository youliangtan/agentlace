#!/usr/bin/env python3

import time
import logging
from agentlace.trainer import TrainerClient, TrainerServer, TrainerConfig
from agentlace.data.data_store import QueuedDataStore, DataStoreBase

import numpy as np
from typing import Any

CLIENT_CAPACITY = 3
SERVER_CAPACITY = 6
IS_REPLAY_BUFFER = False  # NOTE this is for testing replay buffer

################################################################################


def new_data_callback(table_name, payload: Any) -> dict:
    """Simulated callback for training data."""
    assert table_name in {"table1", "table2"}, f"Invalid table name"
    return {}


def request_callback(type: str, payload: Any) -> dict:
    """Simulated callback for getting stats."""
    assert type == "get-stats", "Invalid request type"
    return {"trainer-status": "it's working"}


def helper_create_data_store(capacity) -> DataStoreBase:
    """Helper function to create a data store."""
    if IS_REPLAY_BUFFER:
        from agentlace.data.trajectory_buffer import DataShape
        from agentlace.data.jaxrl_data_store import TrajectoryBufferDataStore
        ds = TrajectoryBufferDataStore(
            capacity=capacity,
            data_shapes=[DataShape(name="index", shape=(3,), dtype="int32")]
        )
    else:
        ds = QueuedDataStore(capacity)
    return ds


def insert_helper(ds: DataStoreBase, data: Any):
    if IS_REPLAY_BUFFER:
        ds.insert({"index": data}, end_of_trajectory=False)
    else:
        ds.insert(data)

################################################################################


def test_queued_data_store():
    ds = QueuedDataStore(6)

    assert len(ds) == 0
    ds.insert(1)
    ds.insert(2)
    assert len(ds) == 2

    data_id_2 = ds.latest_data_id()
    assert len(ds.get_latest_data(data_id_2)) == 0

    ds.insert(3)
    assert len(ds) == 3
    assert len(ds.get_latest_data(data_id_2)) == 1

    ds.batch_insert([4, 5, 6])
    assert len(ds) == 6
    data_id_6 = ds.latest_data_id()

    ds.batch_insert([7, 8, 9])
    assert len(ds) == 6
    # 3 is lost since the capacity is 6
    assert ds.get_latest_data(data_id_2) == [4, 5, 6, 7, 8, 9]
    assert ds.get_latest_data(data_id_6) == [7, 8, 9]

    # this checks if batch insert will only insert the last n=6 data
    # since the capacity is 6
    data_id_9 = ds.latest_data_id()
    ds.batch_insert([10, 11, 12, 13, 14, 15, 16, 17])
    assert ds._data_queue[-1] == 17


def test_trainer():
    # receive network broadcast callback
    received_broadcasted_network = None

    def _network_received_callback(payload: dict):
        """Callback for when client receives weights."""
        nonlocal received_broadcasted_network
        received_broadcasted_network = payload
        print("Client received updated network:", payload)

    # 1. Set up Trainer Server
    trainer_config = TrainerConfig(
        port_number=5555,
        broadcast_port=5556,
        request_types=["get-stats"],
        # experimental_pipeline_port=5547,
    )
    server = TrainerServer(trainer_config, new_data_callback, request_callback)

    # register a data store to trainer server with 2 data stores
    ds_trainer1 = helper_create_data_store(SERVER_CAPACITY)
    ds_trainer2 = helper_create_data_store(SERVER_CAPACITY)
    server.register_data_store("table1", ds_trainer1)
    server.register_data_store("table2", ds_trainer2)
    server.start(threaded=True)

    assert len(ds_trainer1) == 0, "Invalid server data store length"
    assert "table1" in server.store_names(), "Invalid table names in server"

    time.sleep(1)  # Give it a moment to start

    # 2. Set up Trainer Client with 2 data stores
    ds_actor1 = helper_create_data_store(CLIENT_CAPACITY)
    ds_actor2 = helper_create_data_store(CLIENT_CAPACITY)
    ds_actor2 = helper_create_data_store(CLIENT_CAPACITY)
    client = TrainerClient(
        'table1',
        '127.0.0.1',
        trainer_config,
        # data_store=ds_actor1,
        data_stores={"table1": ds_actor1, "table2": ds_actor2},
    )
    client.recv_network_callback(_network_received_callback)

    data_point1 = np.array([1, 2, 3])
    insert_helper(ds_actor1, data_point1)
    assert len(ds_actor1) == 1, "Invalid client data store length"

    # 3. Client update the server
    insert_helper(ds_actor1, np.array([4, 5, 6]))

    res = client.update()
    assert res, "Client update failed"
    assert len(ds_actor1) == 2, "Invalid client data store length"
    time.sleep(1)  # Give it a moment to send

    assert len(
        ds_trainer1) == 2, f"Invalid server data store length {len(ds_trainer1)}"

    # 4 More tests on insertions and queues
    insert_helper(ds_actor1, np.array([7, 8, 9]))
    insert_helper(ds_actor2, np.array([30, 31, 32]))
    insert_helper(ds_actor2, np.array([30, 31, 32]))

    assert len(ds_trainer1) == 2
    assert len(ds_trainer2) == 0
    res = client.update()
    assert res, "Client update failed"
    time.sleep(1)  # Give it a moment to send
    assert len(ds_trainer1) == 3
    assert len(ds_trainer2) == 2

    insert_helper(ds_actor1, np.array([10, 11, 12]))
    assert len(ds_actor1) == CLIENT_CAPACITY
    res = client.update()  # explicitly call update
    assert res is not None
    assert len(ds_trainer1) == 4
    assert len(ds_trainer2) == 2

    # 5. Custom get stats request
    response = client.request("get-stats", {})
    assert response['trainer-status'] == "it's working", "Invalid response"
    response = client.request("invalid", {})
    assert response is None, "Invalid request should return None"

    # 6. start a async update operation
    client.start_async_update(interval=0.5)
    insert_helper(ds_actor1, np.array([13, 14, 15]))
    latest_data_id = ds_actor1.latest_data_id()
    time.sleep(1)
    assert ds_trainer1.latest_data_id() == latest_data_id, "Async update failed"

    # 7. Server publishes weights (for this test, just echo the training data)
    network = np.array([100, 200, 300, 400])
    server.publish_network(network)
    time.sleep(1)  # Give it a moment to receive broadcast
    assert received_broadcasted_network is not None, "Client did not receive broadcast"
    assert np.array_equal(received_broadcasted_network, network)

    # 8. Clean up
    # server.save_data("test.pkl")
    print("stopping")
    client.stop()
    server.stop()
    del client
    del server

    print("[test_trainer] All tests passed!\n")


def stress_test_trainer():
    # 1. Set up Trainer Server
    trainer_config = TrainerConfig(
        port_number=5567,
        broadcast_port=5568,
        # NOTE: use pipe for faster datastore update
        # show that speed up from 0.06 to 0.005 sec in stress test
        # experimental_pipeline_port=5547,
    )
    curr_timeout = 5
    server = TrainerServer(trainer_config, new_data_callback, request_callback)
    ds_learner1 = helper_create_data_store(100000)
    ds_learner2 = helper_create_data_store(100000)
    server.register_data_store("table1", ds_learner1)
    server.register_data_store("table2", ds_learner2)
    server.start(threaded=True)

    time.sleep(1)  # Give it a moment to start

    # 2. Set up Trainer Client with single
    ds_actor1 = helper_create_data_store(100000)
    ds_actor2 = helper_create_data_store(100000)

    client = TrainerClient(
        'table1',
        '127.0.0.1',
        trainer_config,
        # data_store=ds_actor1,
        data_stores={"table1": ds_actor1, "table2": ds_actor2},
        timeout_ms=curr_timeout,  # explicitly set a low timeout
    )
    print("Start trainer stress test")

    # 3. Stress test
    for i in range(1000):
        insert_helper(ds_actor1, np.array([i]*300))
        insert_helper(ds_actor2, np.array([i]*100))

        if (i + 1) % 100 == 0:
            start_time = time.time()
            res = client.update()
            print(f"Update time: {time.time() - start_time}")
            print(f" Datastore 1 actor vs learner: {len(ds_actor1)} vs {len(ds_learner1)}")
            print(f" Datastore 2 actor vs learner: {len(ds_actor2)} vs {len(ds_learner2)}")

            assert len(ds_learner1) <= len(ds_actor1), \
                "Data store in learner should be smaller than actor even when there's msg drop"
            assert len(ds_learner2) <= len(ds_actor2), \
                "Data store in learner should be smaller than actor even when there's msg drop"

            # runtime reconfig client timeout
            # Slowly increase the timeout to simulate improving network conditions
            curr_timeout += 8
            client.req_rep_client.timeout_ms = curr_timeout
            client.req_rep_client.reset_socket()

        time.sleep(0.01)

    print(f"~Final~ DataStore 1 actor vs learner: {len(ds_actor1)} vs {len(ds_learner1)}")
    print(f"~Final~ DataStore 2 actor vs learner: {len(ds_actor2)} vs {len(ds_learner2)}")
    assert len(ds_actor1) == len(ds_learner1), "both ds should have the same length"
    assert len(ds_actor2) == len(ds_learner2), "both ds should have the same length"
    client.stop()
    server.stop()
    del client
    del server
    print("[stress_test_trainer] All tests passed!\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_queued_data_store()
    test_trainer()
    stress_test_trainer()
