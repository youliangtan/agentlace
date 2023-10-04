import logging

import jax
import flax
import chex

import gym
import tqdm

from common import make_agent

from jaxrl_m.agents.continuous.actor_critic import ActorCriticAgent
from jaxrl_m.common.typing import Params

from edgeml.trainer import TrainerClient, TrainerConfig
from edgeml.data.replay_buffer import ReplayBuffer, DataShape


def main():
    # Make the environment
    env = gym.make("HalfCheetah-v4")

    # Make the replay bufffer
    replay_buffer = ReplayBuffer(
        capacity=100000,
        data_shapes=[
            DataShape("observations", env.observation_space.shape),
            DataShape("actions", env.action_space.shape),
            DataShape("rewards", ()),
            DataShape("terminals", (), dtype="bool"),
            DataShape("next_observations", env.observation_space.shape),
        ],
    )

    # Make an EdgeML client
    client = TrainerClient(
        "train",
        "localhost",
        TrainerConfig(),
        replay_buffer,
        log_level=logging.WARNING,
    )

    # Make the agent
    agent: ActorCriticAgent = make_agent(env.observation_space.sample()[None], env.action_space.sample()[None])

    # Training loop
    obs, _ = env.reset()
    action_seed = jax.random.PRNGKey(0)

    def update_params(params: Params):
        nonlocal agent
        chex.assert_trees_all_equal_shapes(params, agent.state.params)
        agent = agent.replace(state=agent.state.replace(params=params))

    client.start_async_update(1)
    client.recv_network_callback(update_params)

    for step in tqdm.trange(1000000):
        # Sample action
        if step < 1000:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(obs, seed=action_seed)

        # Step the environment
        next_obs, reward, done, truncated, _ = env.step(action)
        replay_buffer.insert(
            dict(
                observations=obs,
                actions=action,
                rewards=reward,
                terminals=done and not truncated,
                next_observations=next_obs,
            ),
            end_of_trajectory=done or truncated,
        )

        obs = next_obs
        
        # client.update()  # or update the trainer's datastore

        # Handle termination
        if done or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
