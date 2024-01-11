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
    assert table_name == "table1", "Invalid table name"
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
    assert ds.latest_data_id() - data_id_9 == 6


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
    )
    server = TrainerServer(trainer_config, new_data_callback, request_callback)

    # register a data store to trainer server
    ds_server = helper_create_data_store(SERVER_CAPACITY)
    server.register_data_store("table1", ds_server)
    server.start(threaded=True)

    assert len(ds_server) == 0, "Invalid server data store length"
    assert "table1" in server.store_names(), "Invalid table names in server"

    time.sleep(1)  # Give it a moment to start

    # 2. Set up Trainer Client
    ds_client = helper_create_data_store(CLIENT_CAPACITY)
    client = TrainerClient(
        'table1', '127.0.0.1', trainer_config, data_store=ds_client)
    client.recv_network_callback(_network_received_callback)

    data_point1 = np.array([1, 2, 3])
    insert_helper(ds_client, data_point1)
    assert len(ds_client) == 1, "Invalid client data store length"

    len(client.data_store) == 1, "Invalid client data store length"

    # 3. Client update the server
    insert_helper(ds_client, np.array([4, 5, 6]))

    res = client.update()
    assert res, "Client update failed"
    assert res['success'], "Client update failed"
    assert len(ds_client) == 2, "Invalid client data store length"
    time.sleep(1)  # Give it a moment to send

    assert len(ds_server) == 2, f"Invalid server data store length {len(ds_server)}"

    # 4 More tests on insertions and queues
    insert_helper(ds_client, np.array([7, 8, 9]))

    assert len(ds_server) == 2
    client.update()  # assume client update is successful
    assert len(ds_server) == 3

    insert_helper(ds_client, np.array([10, 11, 12]))
    assert len(ds_client) == CLIENT_CAPACITY
    client.update() # explicitly call update
    assert len(ds_server) == 4

    # 5. Custom get stats request
    response = client.request("get-stats", {})
    assert response['trainer-status'] == "it's working", "Invalid response"
    response = client.request("invalid", {})
    assert response is None, "Invalid request should return None"

    # 6. start a async update operation
    client.start_async_update(interval=0.5)
    insert_helper(ds_client, np.array([13, 14, 15]))
    latest_data_id = ds_client.latest_data_id()
    time.sleep(1)
    assert ds_server.latest_data_id() == latest_data_id, "Async update failed"

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_queued_data_store()
    test_trainer()
