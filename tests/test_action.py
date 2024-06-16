# !/usr/bin/env python3

import cv2
import time
import logging
from agentlace.action import ActionClient, ActionServer, ActionConfig

def test_action():
    # 1. Read the image using OpenCV
    img = cv2.imread("agentlace/tests/test_image.png")

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
    config = ActionConfig(
        port_number=5588,
        action_keys=['send_image'],
        observation_keys=['test_image'],
        broadcast_port=5589,
    )

    # Initialize and start the ActionServer in a separate thread
    server = ActionServer(config, obs_callback, act_callback)
    server.start(threaded=True)

    # Give the server a moment to start up
    time.sleep(2)

    # Initialize the ActionClient
    client = ActionClient('127.0.0.1', config)

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
    print("[test_action] Cleaning up...")
    server.stop()
    client.stop()
    del server
    del client

    print("[test_action] All tests passed!")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_action()
