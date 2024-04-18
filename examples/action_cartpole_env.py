#!/usr/bin/env python

from agentlace.action_env import ActionServerEnvWrapper, ActionClientEnv
import gym
import argparse

"""
This example demonstrates how to wrap a simple gym env to make it distributed.

for example:
    env = gym.make('CartPole-v1')
    
We would want to wrap it as such that we can run the env as server and client.

1. Run the server:
    python example_env_service.py --server
    This wraps the env = ActionServerEnvWrapper(env)
    
2. Run the client:
    python example_env_service.py --client
    Use a generic client to interact with the server.
"""

def run_gym_env(env: gym.Env):

    print(env.observation_space)
    print(env.action_space)

    for episode in range(10):
        observation = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()  # Randomly select an action
            observation, reward, done, trunc, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode+1}: Total Reward = {total_reward}")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--render_mode", type=str, default="human")
    args = parser.parse_args()

    if args.server:
        print("Running Action Env server")
        env = gym.make('CartPole-v1', render_mode=args.render_mode)
        env = ActionServerEnvWrapper(env, port=args.port)
        env.start()
        env.stop()
        print("Server stopped")

    elif args.client:
        print("Running Action Env client")
        env = ActionClientEnv(host=args.host, port=args.port)
        run_gym_env(env)

    else:
        print("Running default cartpole env")
        env = gym.make('CartPole-v1', render_mode=args.render_mode)
        run_gym_env(env)

    print("Done")
