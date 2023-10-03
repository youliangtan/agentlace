import jax
import gym
from jaxrl_m.agents.continuous.actor_critic import ActorCriticAgent

def make_agent(sample_obs, sample_action):
    return ActorCriticAgent.create_states(
        jax.random.PRNGKey(0),
        sample_obs,
        sample_action,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parametrization": "uniform",
        },
    )