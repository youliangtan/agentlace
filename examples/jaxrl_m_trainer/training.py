import time
import logging

import jax
import flax
import chex

import tqdm

from jaxrl_m.agents.continuous.actor_critic import ActorCriticAgent
from jaxrl_m.common.wandb import WandBLogger

from edgeml.trainer import TrainerServer, TrainerConfig
from edgeml.data.replay_buffer import ReplayBuffer, DataShape
from edgeml.data.sampler import LatestSampler

from common import make_agent

def main():
    data_store: ReplayBuffer = ReplayBuffer(
        capacity=100000,
        data_shapes=[
            DataShape("observations", (17,)),
            DataShape("actions", (6,)),
            DataShape("rewards", ()),
            DataShape("terminals", (), dtype="bool"),
            DataShape("next_observations", (17,)),
        ],
    )
    data_store.register_sample_config(
        "train",
        {
            "observations": LatestSampler(),
            "actions": LatestSampler(),
            "rewards": LatestSampler(),
            "terminals": LatestSampler(),
            "next_observations": LatestSampler(),
        },
    )

    # Create server
    server = TrainerServer(TrainerConfig(), log_level=logging.WARNING)
    server.register_data_store("train", data_store)

    # Make an agent
    agent = make_agent(
        data_store.dataset["observations"][:1],
        data_store.dataset["actions"][:1],
    )

    server.start(threaded=True)

    # Wait for enough data
    data_pbar = tqdm.trange(1000, desc="Waiting for data")
    while len(data_store) < 1000:
        data_pbar.update(len(data_store) - data_pbar.n)
        time.sleep(1)

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

    # Training loop
    import itertools
    for train_step in tqdm.tqdm(itertools.count(), desc="Training"):
        # Sample from RB
        batch = data_store.sample(128)

        # Update agent
        agent, update_info = agent.update(batch)

        # Logging
        if train_step % 100 == 0:
            wandb_logger.log(
                update_info,
                step=train_step,
            )

        # Update params
        server.publish_network(agent.state.params)

if __name__ == "__main__":
    main()