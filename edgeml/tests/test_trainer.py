#!/usr/bin/env python3

import time
import logging
from edgeml.interfaces import TrainerClient, TrainerServer, TrainerConfig, DataTable

def dummy_training_callback(table_name, payload: dict) -> dict:
    """Simulated callback for training data."""
    print("Server received training data:", payload, " for table:", table_name)
    # For the sake of this test, just echo back the data as "weights"
    return {"weights": payload['data']}

def request_callback(type: str, payload: dict) -> dict:
    """Simulated callback for getting stats."""
    assert type == "get-stats", "Invalid request type"
    return {"trainer-status": "it's working"}

def test_trainer():
    # receive weights broadcast callback
    received_broadcast = False
    def weights_received_callback(payload: dict):
        """Callback for when client receives weights."""
        nonlocal received_broadcast
        received_broadcast = True
        print("Client received weights:", payload)

    # 1. Set up Trainer Server
    # try make from file method
    server = TrainerServer.make_from_file(
        "invalid.pkl", dummy_training_callback, request_callback)
    assert server is None, "Invalid file should return None"

    server_config = TrainerConfig(
            port_number=5555,
            broadcast_port=5556,
            data_table=[DataTable(name="table1", size=2),
                        DataTable(name="table2", size=3)],
            request_types=["get-stats"],
        )
    server = TrainerServer(server_config, dummy_training_callback, request_callback)
    server.start(threaded=True)
    assert len(server.table_names()) == 2, "Invalid table names in server"
    time.sleep(1)  # Give it a moment to start

    # 2. Set up Trainer Client
    client = TrainerClient('127.0.0.1', server_config)
    client.recv_weights_callback(weights_received_callback)

    # 3. Client sends dummy training data
    training_payload = {"data": [1, 2, 3, 4, 5], "time": time.time()}
    response = client.train_step("table1", training_payload)
    print("Client received response:", response)

    # 4. Custom get stats request
    response = client.request("get-stats", {})
    print("Client received response:", response)
    assert response['trainer-status'] == "it's working", "Invalid response"
    
    response = client.request("invalid", {})
    assert response is None, "Invalid request should return None"

    # 5. Server publishes weights (for this test, just echo the training data)
    server.publish_weights({"weights": training_payload['data']})
    time.sleep(1)  # Give it a moment to receive broadcast
    assert received_broadcast, "Client did not receive broadcasted weights"

    # 6. Server is able to queue up to the max queue size
    training_payload = {"data": [6, 7, 8], "time": time.time()}
    response = client.train_step("table1", training_payload)
    training_payload = {"data": [9, 10], "time": time.time()}
    response = client.train_step("table1", training_payload)
    datas = server.get_data("table1")
    print("Server received data:", datas)
    assert len(datas) == 2, "Server should have 2 data points in queue"
    assert datas[0]['data'] == [6, 7, 8], "Server queue should be FIFO"

    # 7. check data table size
    datas = server.get_data("table2")
    print("Server received data:", server.data_store)
    assert len(datas) == 0, "Server should have no data points in queue"
    assert server.get_data("table99") is None, "Invalid table name should return None"

    # 8. Clean up
    # server.save_data("test.pkl")
    server.stop()
    client.stop()

    print("[test_trainer] All tests passed!")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_trainer()
