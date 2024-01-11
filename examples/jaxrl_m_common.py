# !/usr/bin/env python3

import jax
import chex
import gym
import numpy as np
from typing import Optional

from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.common.wandb import WandBLogger

from agentlace.trainer import TrainerConfig
from agentlace.data.trajectory_buffer import DataShape
from agentlace.data.jaxrl_data_store import TrajectoryBufferDataStore
from agentlace.data.sampler import LatestSampler, SequenceSampler

from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.agents.continuous.sac import SACAgent
from agentlace.trainer import TrainerConfig

from jax import nn
from oxe_envlogger.rlds_logger import RLDSLogger


##############################################################################


def make_agent(sample_obs, sample_action):
    return SACAgent.create_states(
        jax.random.PRNGKey(0),
        sample_obs,
        sample_action,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "softplus",
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=0.99,
        backup_entropy=True,
        critic_ensemble_size=10,
        critic_subsample_size=2,
    )


def make_trainer_config():
    return TrainerConfig(
        port_number=5488,
        broadcast_port=5489,
        request_types=["send-stats"]
    )


def make_wandb_logger(
    project: str = "agentlace",
    description: str = "jaxrl_m",
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
    )
    return wandb_logger
