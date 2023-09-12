# !/usr/bin/env python3

import argparse
from edgeml.interfaces import EdgeClient, EdgeServer, EdgeConfig
import cv2
import time

cap = cv2.VideoCapture(0)

def obs_callback(keys, payload):
    print("Observation requested from client: ", keys)
    start = time.time()
    # TODO get image from webcam
    # img = cv2.imread("edgeml/tests/test_image.png")    
    # Capture image from webcam
    ret, img = cap.read()

    if not ret:
        print("Error capturing image from webcam.")
        return {'image': None}

    obj = {"image": img}
    # print("Time taken to capture image: ", time.time() - start)
    return obj

def act_callback(key, payload):
    print("action requested from client! ", key)
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

    config = EdgeConfig(
        port_number = args.port,
        action_keys = ["init", "move", "gripper", "reset", "start"],
        observation_keys = ["image", "proprio"],
    )

    if args.server:
        server = EdgeServer(config, obs_callback=obs_callback, act_callback=act_callback)
        server.start()

    if args.client:
        client = EdgeClient(args.ip, config)
        # 5Hz get image from server and display
        end_time = time.time() + 10  # For example, run for 10 seconds
        while time.time() < end_time:
            start = time.time()
            obs = client.obs()
            img = obs["image"]
            if img is None:
                continue
            
            # # assert img is not None
            # # assert img.shape == (256, 256, 3)
            cv2.imshow("image", img)
            if cv2.waitKey(100) & 0xFF == ord('q'):  # Wait for 200 ms; quit on 'q' keypress
                break
            
            print("Time taken to get image: ", time.time() - start)

        cv2.destroyAllWindows()

        # TODO: 5 hz get image from server and display
    
        # obs = client.obs()
        # img = obs["image"]
        # assert img is not None
        # assert img.shape == (256, 256, 3)
        # cv2.imshow("image", img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # time.sleep(0.2)  # 5Hz
        # # cv2.imshow("image", img)
        # # cv2.waitKey(0)

        res = client.act("move")
        print(res)

    print("Done")
