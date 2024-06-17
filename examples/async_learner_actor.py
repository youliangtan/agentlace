#!/usr/bin/env python3

# NOTE: this requires jaxrl_m to be installed:
#       https://github.com/rail-berkeley/jaxrl_minimal
# Requires mujoco_py and mujoco==2.2.2

import time
from functools import partial

import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.common.evaluation import evaluate
from jaxrl_m.utils.timer_utils import Timer

from agentlace.trainer import TrainerServer, TrainerClient, TrainerSMInterface
from agentlace.data.data_store import QueuedDataStore
from agentlace.data.jaxrl_data_store import ReplayBufferDataStore
from agentlace.data.jaxrl_data_store import make_default_trajectory_buffer

from jaxrl_m_common import make_agent, make_trainer_config, make_wandb_logger

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "HalfCheetah-v4", "Name of environment.")
flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 1000, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("utd_ratio", 8, "UTD ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 1000000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 500, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 1000, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 10000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")

# experimental with trajectory buffer
flags.DEFINE_boolean("use_traj_buffer", False, "Use efficient replay buffer.")

# save replaybuffer data as rlds tfrecord
flags.DEFINE_string("rlds_log_dir", None, "Directory to log data.")


def print_green(x): return print("\033[92m {}\033[00m" .format(x))


def print_yellow(x): return print("\033[93m {}\033[00m" .format(x))

##############################################################################


global_rlds_logger = None


def actor(agent: SACAgent, data_store, env, sampling_rng, sm_trainer=None):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    NOTE: sm_trainer is used the transport layer for multi-threading
    """
    if sm_trainer:
        client = sm_trainer
    else:
        client = TrainerClient(
            "actor_env",
            FLAGS.ip,
            make_trainer_config(),
            data_store,
            wait_for_server=True,
        )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        # chex.assert_trees_all_equal_shapes(params, agent.state.params)
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    eval_env = gym.make(FLAGS.env)
    eval_env = RecordEpisodeStatistics(eval_env)

    obs, _ = env.reset()
    done = False

    # NOTE: either use client.update() or client.start_async_update()
    # client.start_async_update(interval=1)  # every 1 sec

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
            info = np.asarray(info)
            running_return += reward

            data_payload = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
            )

            if FLAGS.use_traj_buffer:
                # NOTE: end_of_trajectory is used in TrajectoryBuffer
                data_payload["end_of_trajectory"] = truncated # TODO: check if ignore None is okay

            data_store.insert(data_payload)

            obs = next_obs
            if done or truncated:
                running_return = 0.0
                obs, _ = env.reset()

        if FLAGS.render:
            env.render()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            client.request("send-stats", stats)

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)

##############################################################################


def learner(agent, replay_buffer, wandb_logger=None, sm_trainer=None, sharding=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    NOTE: sm_trainer is used the transport layer for multi-threading
    """
    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    if sm_trainer:
        sm_trainer.register_request_callback(stats_callback)
        server = sm_trainer
    else:
        server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
        server.register_data_store("actor_env", replay_buffer)
        server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(total=FLAGS.training_starts, initial=len(replay_buffer),
                     desc="Filling up replay buffer", position=0, leave=True)
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green('sent initial network to actor')

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # Train the networks
        with timer.context("sample_replay_buffer"):
            if FLAGS.use_traj_buffer:
                batch, mask = replay_buffer.sample(
                    "training",  # define in the TrajectoryBuffer.register_sample_config
                    FLAGS.batch_size,
                )
                # replay_buffer's batch is default in cpu, put it to devices
                batch = jax.device_put(batch, sharding.replicate())
            else:
                batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)

        with timer.context("train"):
            agent, update_info = agent.update_high_utd(
                batch, utd_ratio=FLAGS.utd_ratio
            )
            agent = jax.block_until_ready(agent)

            # publish the updated network
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)
        update_steps += 1

##############################################################################


def create_datastore_and_wandb_logger(env):
    """
    Utility function to create replay buffer and wandb logger.
    """
    if FLAGS.rlds_log_dir:
        print_yellow(f"Saving replay buffer data to {FLAGS.rlds_log_dir}")
        # Install from: https://github.com/rail-berkeley/oxe_envlogger
        from oxe_envlogger.rlds_logger import RLDSLogger

        logger = RLDSLogger(
            observation_space=env.observation_space,
            action_space=env.action_space,
            dataset_name=FLAGS.env,
            directory=FLAGS.rlds_log_dir,
            max_episodes_per_file=10,  # TODO: arbitrary number
        )
        global global_rlds_logger
        global_rlds_logger = logger

    if FLAGS.use_traj_buffer:
        print_yellow(f"Using experimental Trajectory buffer")
        replay_buffer = make_default_trajectory_buffer(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger=logger if FLAGS.rlds_log_dir else None,
        )
    else:
        replay_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger=logger if FLAGS.rlds_log_dir else None,
        )

    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="jaxrl_minimal",
        description=FLAGS.exp_name or FLAGS.env,
    )
    return replay_buffer, wandb_logger


##############################################################################

def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % num_devices == 0

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    if FLAGS.render:
        env = gym.make(FLAGS.env, render_mode="human")
    else:
        env = gym.make(FLAGS.env)

    rng, sampling_rng = jax.random.split(rng)
    agent: SACAgent = make_agent(
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: SACAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.learner:
        replay_buffer, wandb_logger = create_datastore_and_wandb_logger(env)

        # learner loop
        print_green("starting learner loop")
        learner(agent, replay_buffer, wandb_logger=wandb_logger, sharding=sharding)

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(5000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng, sm_trainer=None)

    else:
        print_green("starting actor and learner loop with multi-threading")

        # Here, the shared-memory interface acts as the transport layer for the
        # trainerServer and trainerClient. Also, both actor and learner shares
        # the same replay buffer.
        replay_buffer, wandb_logger = create_datastore_and_wandb_logger(env)

        sm_trainer = TrainerSMInterface()
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())

        import threading
        # Start learner thread
        learner_thread = threading.Thread(
            target=learner,
            args=(agent, replay_buffer, wandb_logger, sm_trainer, sharding)
        )
        learner_thread.start()

        # Start actor in main process
        actor(agent, replay_buffer, env, sampling_rng, sm_trainer=sm_trainer)
        learner_thread.join()


if __name__ == "__main__":
    try:
        app.run(main)
    finally:
        # NOTE: manually flush the logger when exit to prevent data loss
        # this is required as the envlogger writer doesn't handle
        # destruction of the object gracefully
        if global_rlds_logger:
            global_rlds_logger.close()
        print_green("done exit")
