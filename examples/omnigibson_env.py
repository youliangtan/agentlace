"""
Example of using: https://github.com/StanfordVL/OmniGibson

Ref:
    https://github.com/StanfordVL/OmniGibson/blob/main/omnigibson/examples/learning/navigation_policy_demo.py

To run:
1. Start the server: python omnigibson_env.py
2. Start the client: python omnigibson_env.py --client
"""

import argparse
import yaml

import omnigibson as og
from omnigibson import example_config_path
from omnigibson.macros import gm
from omnigibson.utils.python_utils import meets_minimum_version
import gymnasium as gym

from agentlace.gym_env import GymEnvServerWrapper, GymEnvClient
import cv2
import numpy as np


assert meets_minimum_version(gym.__version__, "0.28.1"), "Please install/update gymnasium to version >= 0.28.1"

# We don't need object states nor transitions rules, so we disable them now, and also enable flatcache for maximum speed
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_FLATCACHE = True

def run_env(env):
    print("Running environment")
    # reset the environment
    print("Observation space:", env.observation_space)
    image_obs_key = list(env.observation_space.spaces.keys())[0]

    obs, info = env.reset()

    for i in range(1000):
        print("step", i)
        # action = np.random.randn(env.robots[0].dof)  # sample random action
        action = env.action_space.sample()
        print(action)
        obs, reward, done, trunc, info = env.step(action)  # take action in the environment

        # convert tensor to numpy array, orginal shape is torch.Size([128, 128, 4])
        image = obs[image_obs_key].numpy()
        # remove the last channel
        image = image[:, :, :3]
        # Convert the image from RGB to BGR (OpenCV uses BGR by default)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the image
        cv2.imshow("Agent View", image)

        # Press 'q' to quit the display window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        env.render()  # render on display

    # Release the display window
    cv2.destroyAllWindows()

def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()
    
    # if GymEnv client
    if args.client:
        print("Running Action Env client")
        env = GymEnvClient(host=args.host, port=args.port, timeout_ms=1500)
        run_env(env)
        return

    # Load config
    with open(f"{example_config_path}/turtlebot_nav.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Make sure flattened obs and action space is used
    cfg["env"]["flatten_action_space"] = True
    cfg["env"]["flatten_obs_space"] = True

    # Only use RGB obs
    cfg["robots"][0]["obs_modalities"] = ["rgb"]

    # If we're not eval, turn off the start / goal markers so the agent doesn't see them
    if not args.eval:
       cfg["task"]["visualize_goal"] = False

    env = og.Environment(configs=cfg)

    # If we're evaluating, hide the ceilings and enable camera teleoperation so the user can easily
    # visualize the rollouts dynamically
    if args.eval:
        ceiling = env.scene.object_registry("name", "ceilings")
        ceiling.visible = False
        og.sim.enable_viewer_camera_teleoperation()

    env = GymEnvServerWrapper(env, port=args.port)
    # env.reset()
    env.start()
    env.stop()
    print("Server stopped")
    print("done")


if __name__ == "__main__":
    main()
