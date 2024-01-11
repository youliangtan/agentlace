from functools import partial

import chex
import gym
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from flax import linen as nn
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.common.evaluation import evaluate

from agentlace.data.trajectory_buffer import TrajectoryBuffer, DataShape
from agentlace.data.sampler import LatestSampler, SequenceSampler

from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m_common import make_agent, make_wandb_logger

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "HalfCheetah-v4", "Name of environment.")
flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_string("save_dir", "jaxrl_log", "Logging dir.")
flags.DEFINE_integer("max_traj_length", 1000, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 1024, "Batch size.")
flags.DEFINE_integer("utd_ratio", 8, "UTD ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 1000000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 500, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 1000, "Training starts after this step.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 10000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

flags.DEFINE_bool(
    "deterministic_eval", True, "Whether to use deterministic policy for evaluation."
)
flags.DEFINE_bool("debug", False, "Debug config")


def main(_):
    # devices = jax.local_devices()
    # num_devices = len(devices)
    # sharding = jax.sharding.PositionalSharding(devices)
    # assert FLAGS.batch_size % num_devices == 0

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="jaxrl_minimal_agentlace",
        description=FLAGS.exp_name or FLAGS.env
    )

    # create env and load dataset
    env = gym.make(FLAGS.env)
    eval_env = gym.make(FLAGS.env)
    eval_env = RecordEpisodeStatistics(eval_env)

    rng, sampling_rng = jax.random.split(rng)
    agent: SACAgent = make_agent(
        sample_obs=env.observation_space.sample()[None],
        sample_action=env.action_space.sample()[None],
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    # agent: SACAgent = jax.device_put(
    #     jax.tree_map(jnp.array, agent), sharding.replicate()
    # )

    replay_buffer = TrajectoryBuffer(
        capacity=FLAGS.replay_buffer_capacity,
        data_shapes=[
            DataShape("observations", env.observation_space.shape, np.float32),
            DataShape("next_observations", env.observation_space.shape, np.float32),
            DataShape("actions", env.action_space.shape, np.float32),
            DataShape("rewards", (), np.float32),
            DataShape("masks", (), np.float32),
            DataShape("end_of_trajectory", (), dtype="bool"),
        ],
        min_trajectory_length=2,
    )

    @jax.jit
    def transform_rl_data(batch, mask):
        batch_size = jax.tree_util.tree_flatten(batch)[0][0].shape[0]
        chex.assert_tree_shape_prefix(batch["observations"], (batch_size, 2))
        chex.assert_tree_shape_prefix(mask["observations"], (batch_size, 2))
        return {
            **batch,
            "observations": batch["observations"][:, 0],
            "next_observations": batch["observations"][:, 1],
        }, {
            **mask,
            "observations": mask["observations"][:, 0],
            "next_observations": mask["observations"][:, 1],
        }

    replay_buffer.register_sample_config(
        "training",
        samplers={
            "observations": SequenceSampler(
                squeeze=False, begin=0, end=2, source="observations"
            ),
            "actions": LatestSampler(),
            "rewards": LatestSampler(),
            "masks": LatestSampler(),
            "next_observations": LatestSampler(),
            "end_of_trajectory": LatestSampler(),
        },
        transform=transform_rl_data,
        sample_range=(0, 2),
    )

    # TODO: adding seed to env doesn't work for some reason
    obs, _ = env.reset()

    # training loop
    timer = Timer()
    running_return = 0.0
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)
            done = done
            info = np.asarray(info)
            running_return += reward

            replay_buffer.insert(
                dict(
                    observations=obs,
                    next_observations=next_obs,
                    actions=actions,
                    rewards=reward,
                    masks=1.0 - done,
                    end_of_trajectory=done or truncated,
                ),
                end_of_trajectory=done or truncated,
            )

            obs = next_obs
            if done or truncated:
                running_return = 0.0
                obs, _ = env.reset()

        # Train the networks
        if step >= FLAGS.training_starts:
            with timer.context("sample_replay_buffer"):
                batch, mask = replay_buffer.sample(
                    "training",
                    FLAGS.batch_size,
                )
                batch = jax.device_put(batch, jax.local_devices()[0])
                batch = jax.block_until_ready(batch)

            with timer.context("train"):
                agent, update_info = jax.block_until_ready(
                    agent.update_high_utd(batch, utd_ratio=FLAGS.utd_ratio)
                )

            if step % FLAGS.log_period == 0:
                wandb_logger.log(
                    update_info,
                    step=step,
                )

        if step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            wandb_logger.log(
                {"eval": evaluate_info},
                step=step,
            )

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            wandb_logger.log(
                {
                    "timer": timer.get_average_times(),
                },
                step=step,
            )


if __name__ == "__main__":
    app.run(main)
