# !/usr/bin/env python3

# NOTE: this requires jaxrl_m to be installed: 
#       https://github.com/rail-berkeley/jaxrl_minimal

import jax
import gym

from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.common.wandb import WandBLogger

from edgeml.trainer import TrainerConfig
from edgeml.data.data_store import DataStoreBase

from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.data.replay_buffer import ReplayBuffer
from edgeml.trainer import TrainerConfig

from threading import Lock

from jax import nn

##############################################################################


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


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
    project: str = "edgeml",
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
