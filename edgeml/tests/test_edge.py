# !/usr/bin/env python3


import cv2
import time
import logging
from edgeml.interfaces import EdgeClient, EdgeServer, EdgeConfig

def test_edge():
    # 1. Read the image using OpenCV
    img = cv2.imread("edgeml/tests/test_image.png")

    # Define callback functions for the server
    def obs_callback(keys: set) -> dict:
        if "test_image" in keys:
            return {"status": "success", "test_image": img}
        return {"status": "error", "message": "Invalid key requested"}

    def act_callback(key: str, payload: dict) -> dict:
        if key == 'send_image':
            # Do something with the image, for now, just return a success message
            return {"status": "success", "message": "Image received"}
        return {"status": "error", "message": "Invalid action"}

    # Define our config
    config = EdgeConfig(
        port_number=5555,
        action_keys=['send_image'],
        observation_keys=['test_image'],
        broadcast_port=5556
    )

    # Initialize and start the EdgeServer in a separate thread
    server = EdgeServer(config, obs_callback, act_callback)
    server.start(threaded=True)

    # Give the server a moment to start up
    time.sleep(2)

    # Initialize the EdgeClient
    client = EdgeClient('127.0.0.1', config)

    # Define the callback for the client to handle broadcasted data
    received_broadcast = False
    def client_callback(data):
        nonlocal received_broadcast
        received_broadcast = True
        print(f"Client received data: {data}")
        assert "test-broadcast" in data, "incorrect broadcasted data from client"

    # Register the callback with the client
    client.register_obs_callback(client_callback)
    observation = client.obs()
    print(f"Client received observation")
    img = observation['test_image']
    assert img.shape == (256, 256, 3), "incorrect image shape"

    # Now, let's pretend the server wants to broadcast the image shape
    server.publish_obs({"test-broadcast": 1})

    # Give the client a moment to process the broadcasted data
    time.sleep(2)
    assert received_broadcast, "Client did not receive broadcasted data"

    # Clean up
    server.stop()
    client.stop()

    print("[test_edge] All tests passed!")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_edge()
