import logging

import jax
from flax.training import checkpoints
import chex
import jax.numpy as jnp

import gym
import tqdm
import argparse

from common import make_agent, make_trainer_config, make_wandb_logger

from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.common.typing import Params


from edgeml.trainer import TrainerClient
from edgeml.data.replay_buffer import ReplayBuffer, DataShape
from edgeml.data.sampler import LatestSampler


def main(args):
    # Make the environment
    if args.visualize:
        env = gym.make("HalfCheetah-v4", render_mode="human")
    else:
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
    replay_buffer.register_sample_config(
        "train",
        {
            "observations": LatestSampler(),
            "actions": LatestSampler(),
            "rewards": LatestSampler(),
            "terminals": LatestSampler(),
            "next_observations": LatestSampler(),
            "masks": LatestSampler(),
        },
    )

    # Make the agent
    agent: SACAgent = make_agent(
        env.observation_space.sample()[None],
        env.action_space.sample()[None]
        # replay_buffer.dataset["observations"][:1],
        # replay_buffer.dataset["actions"][:1],
    )

    # Load checkpoint
    if args.load is not None:
        c = checkpoints.restore_checkpoint(args.load, agent.state.params)
        agent = agent.replace(state=agent.state.replace(params=c))
        print(f"Loaded checkpoint from {args.load}")

    # Training loop
    obs, _ = env.reset()
    if args.visualize:
        env.render()

    action_seed = jax.random.PRNGKey(0)

    if args.train:
        wandb_logger = make_wandb_logger()
    else:
        # Make an EdgeML client
        client = TrainerClient(
            "train",
            "localhost",
            make_trainer_config(),
            replay_buffer,
            log_level=logging.WARNING,
            wait_for_server=True,
        )

    def update_params(params: Params):
        nonlocal agent
        chex.assert_trees_all_equal_shapes(params, agent.state.params)
        agent = agent.replace(state=agent.state.replace(params=params))

    # client.start_async_update(1) # TODO
    if not args.train:
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
            if args.train:
                # Train and update agent
                batch, mask = replay_buffer.sample("train", 100, 
                    # NOTE: do i need force indices?
                    # force_indices=jnp.arange(
                    #     replay_buffer._sample_begin_idx, replay_buffer._sample_end_idx
                    # ),
                )
                agent, update_info = agent.update(batch)
                wandb_logger.log(update_info, step=step)
                update_params(agent.state.params)
            else:
                client.update()  # or update the trainer's datastore

        # Handle termination
        if done or truncated:
            stats = dict(ep_return=running_return, ep_len=ep_len)
            # Send stats
            if args.train:
                wandb_logger.log(
                    {f"env/{k}": v for k, v in stats.items()},
                    step=step,
                )
            else:
                client.request("send-stats", stats)

            # Reset
            obs, _ = env.reset()
            running_return = 0
            ep_len = 0
            
        # if step % 30000 == 0:
        #     checkpoints.save_checkpoint(f"checkpoints/c_{step}", agent.state.params, step)
        #     print(f"Saved checkpoint at step {step}")

        if args.visualize:
            env.render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Visualize the environment while training")
    parser.add_argument("--load", type=str, help="Path to checkpoint to load")
    args = parser.parse_args()

    main(args)
