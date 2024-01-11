# !/usr/bin/env python3

# a simple example of the action server capturing image from webcam and
# the client requesting for the image, and displaying it

import argparse
from agentlace.action import ActionClient, ActionServer, ActionConfig
from agentlace.internal.utils import mat_to_jpeg, jpeg_to_mat
import cv2
import time

##############################################################################

# Capture image from webcam
cap = cv2.VideoCapture(0)
CLIENT_TIMEOUT = 8



def obs_callback(keys: set) -> dict:
    """
    this reads the image from the webcam and send it to the client
    """
    print("Observation requested from client: ", keys)
    # img = cv2.imread("agentlace/tests/test_image.png")
    ret, img = cap.read()

    if not ret:
        print("Error capturing image from webcam.")
        return {'image': None}

    # convert to jpeg for compression, but can also send raw image as well
    obj = {"image": mat_to_jpeg(img)}
    return obj


def act_callback(key: str, payload: dict) -> dict:
    print("action requested from client! ", key)
    if key == "move":
        return {"move": "success"}
    return {}

##############################################################################


if __name__ == "__main__":
    # NOTE: This is just for Testing
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--client', action='store_true')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    args = parser.parse_args()

    config = ActionConfig(
        port_number=args.port,
        action_keys=["init", "move", "gripper", "reset", "start"],
        observation_keys=["image", "proprio"],
        broadcast_port=5557
    )

    if args.server:
        server = ActionServer(config,
                              obs_callback=obs_callback,
                              act_callback=act_callback)
        server.start(threaded=True)

        # broadcast observations stream every 1 second
        while True:
            time.sleep(1)
            server.publish_obs({"depth_image": "test-broadcast publish impl"})

    if args.client:
        client = ActionClient(args.ip, config)

        sub_count = 0

        def sub_callback(obs: dict):
            global sub_count
            sub_count += 1
            print("Obs stream: ", obs, sub_count)

        client.register_obs_callback(callback=sub_callback)

        # 5Hz get image from server and display
        end_time = time.time() + CLIENT_TIMEOUT
        while time.time() < end_time:
            start = time.time()
            obs = client.obs()
            print(f" obs took {time.time() - start} seconds")
            img = jpeg_to_mat(obs["image"])

            assert img is not None
            # assert img.shape == (256, 256, 3)

            cv2.imshow("image", img)
            # Wait for 50 ms; quit on 'q' keypress
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

            print("Time taken to get image: ", time.time() - start)

        cv2.destroyAllWindows()
        assert sub_count == CLIENT_TIMEOUT, \
            f"Expected {CLIENT_TIMEOUT} messages, got {sub_count}"
        res = client.act("move")
        assert res["move"] == "success"

    print("Done All")
