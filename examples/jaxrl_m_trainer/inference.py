import logging

import jax
import flax
import chex

import gym
import tqdm

from common import make_agent, make_trainer_config

from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.common.typing import Params

from edgeml.trainer import TrainerClient, TrainerConfig
from edgeml.data.replay_buffer import ReplayBuffer, DataShape


def main():
    # Make the environment
    env = gym.make("HalfCheetah-v4")

    # Make the replay bufffer
    replay_buffer = ReplayBuffer(
        capacity=1000,
        data_shapes=[
            DataShape("observations", env.observation_space.shape),
            DataShape("actions", env.action_space.shape),
            DataShape("rewards", ()),
            DataShape("terminals", (), dtype="bool"),
            DataShape("next_observations", env.observation_space.shape),
            DataShape("masks", ()),
        ],
    )

    # Make an EdgeML client
    client = TrainerClient(
        "train",
        "localhost",
        make_trainer_config(),
        replay_buffer,
        log_level=logging.WARNING,
    )

    # Make the agent
    agent: SACAgent = make_agent(env.observation_space.sample()[None], env.action_space.sample()[None])

    # Training loop
    obs, _ = env.reset()
    action_seed = jax.random.PRNGKey(0)

    def update_params(params: Params):
        nonlocal agent
        chex.assert_trees_all_equal_shapes(params, agent.state.params)
        agent = agent.replace(state=agent.state.replace(params=params))

    # client.start_async_update(1)
    client.recv_network_callback(update_params)

    running_return = 0
    ep_len = 0
    for step in tqdm.trange(100000000):
        # Sample action
        if step < 1000:
            action = env.action_space.sample()
        else:
            action_seed, key = jax.random.split(action_seed)
            action = agent.sample_actions(obs, seed=key)

        # Step the environment
        next_obs, reward, done, truncated, _ = env.step(action)
        replay_buffer.insert(
            dict(
                observations=obs,
                actions=action,
                rewards=reward,
                terminals=done and not truncated,
                next_observations=next_obs,
                masks=(1 - done),
            ),
            end_of_trajectory=done or truncated,
        )
        running_return += reward
        ep_len += 1

        obs = next_obs

        if step % 100 == 0:
            client.update()  # or update the trainer's datastore

        # Handle termination
        if done or truncated:
            # Send stats
            client.request("send-stats", dict(ep_return=running_return, ep_len=ep_len))

            # Reset
            obs, _ = env.reset()
            running_return = 0
            ep_len = 0

if __name__ == "__main__":
    main()
