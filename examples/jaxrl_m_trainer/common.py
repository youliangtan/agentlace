import jax
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.agents.continuous.sac import SACAgent
from edgeml.trainer import TrainerConfig

from jax import nn

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
        request_types=["send-stats"]
    )
    
def make_wandb_logger():
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "edgeml",
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
    )
    return wandb_logger
