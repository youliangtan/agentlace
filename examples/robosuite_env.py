#!/usr/bin/env python
"""
This uses env in https://github.com/ARISE-Initiative/robosuite

# Example:

## normal run:
python robosuite_env.py

## run as server and client
python robosuite_env.py --server
python robosuite_env.py --client

NOTE: make sure https://github.com/ARISE-Initiative/robosuite/pull/497 is merged.
      else type: git pull origin refs/pull/497/head  to use the pending PR branch.
"""

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import cv2
from agentlace.gym_env import GymEnvServerWrapper, GymEnvClient
import argparse


def make_env():

    # create environment instance
    env = suite.make(
        env_name="Lift",  # try with other tasks like "Stack" and "Door"
        robots="Panda",   # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )

    # convert to gym environment
    env = GymWrapper(
        env,
        keys=["robot0_proprio-state", "object-state", "agentview_image"],
        flatten_obs=False
    )
    print(" -> action space:", env.action_space)
    print(" -> observation space:", env.observation_space)
    env.reset()
    return env


def run_env(env):
    print("Running environment")
    # reset the environment
    obs, info = env.reset()

    for i in range(1000):
        print("step", i)
        # action = np.random.randn(env.robots[0].dof)  # sample random action
        action = env.action_space.sample()
        print(action)
        obs, reward, done, trunc, info = env.step(action)  # take action in the environment
        print(obs.keys(), obs["agentview_image"].shape)

        # Convert the image from RGB to BGR (OpenCV uses BGR by default)
        image = cv2.cvtColor(obs["agentview_image"], cv2.COLOR_RGB2BGR)

        # Display the image
        cv2.imshow("Agent View", image)

        # Press 'q' to quit the display window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        env.render()  # render on display

    # Release the display window
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()

    if args.server:
        print("Running Action Env server")
        env = make_env()
        env = GymEnvServerWrapper(env, port=args.port)
        env.start()
        env.stop()
        print("Server stopped")

    elif args.client:
        print("Running Action Env client")
        env = GymEnvClient(host=args.host, port=args.port, timeout_ms=1500)
        run_env(env)

    else:
        print("Running default robosuite env")
        env = make_env()
        run_env(env)

    print("Done")
