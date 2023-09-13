#!/usr/bin/env python3

import time
import logging
from typing import Optional
from edgeml.interfaces import TrainerClient, TrainerServer, TrainerConfig

def dummy_training_callback(payload: dict) -> Optional[dict]:
    """Simulated callback for training data."""
    print("Server received training data:", payload)
    # For the sake of this test, just echo back the data as "weights"
    return {"weights": payload['data']}


def test_trainer():
    
    # receive weights broadcast callback
    received_broadcast = False
    def weights_received_callback(payload: dict):
        """Callback for when client receives weights."""
        nonlocal received_broadcast
        received_broadcast = True
        print("Client received weights:", payload)

    # 1. Set up Trainer Server
    server_config = TrainerConfig(
            port_number=5555,
            broadcast_port=5556,
            queue_size=2
        )
    server = TrainerServer(server_config, dummy_training_callback)
    server.start(threaded=True)
    time.sleep(1)  # Give it a moment to start

    # 2. Set up Trainer Client
    client = TrainerClient('127.0.0.1', server_config)
    client.recv_weights_callback(weights_received_callback)

    # 3. Client sends dummy training data
    training_payload = {"data": [1, 2, 3, 4, 5], "time": time.time()}
    response = client.train_step(training_payload)
    print("Client received response:", response)

    # 4. Server publishes weights (for this test, just echo the training data)
    server.publish_weights({"weights": training_payload['data']})
    time.sleep(1)  # Give it a moment to receive broadcast
    assert received_broadcast, "Client did not receive broadcasted weights"

    # 5. Server is able to queue up to the max queue size
    training_payload = {"data": [6, 7, 8], "time": time.time()}
    response = client.train_step(training_payload)
    training_payload = {"data": [9, 10], "time": time.time()}
    response = client.train_step(training_payload)
    datas = server.get_data()
    assert len(datas) == 2, "Server should have 2 data points in queue"
    assert datas[0]['data'] == [6, 7, 8], "Server queue should be FIFO"

    # 5. Clean up
    server.stop()
    client.stop()

    print("[test_trainer] All tests passed!")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_trainer()
