import time
import logging

import jax
import flax
import chex

import tqdm
import itertools

from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.common.wandb import WandBLogger

from edgeml.trainer import TrainerServer, TrainerConfig
from edgeml.data.replay_buffer import ReplayBuffer, DataShape
from edgeml.data.sampler import LatestSampler

from common import make_agent, make_trainer_config, make_wandb_logger


def main():
    data_store: ReplayBuffer = ReplayBuffer(
        capacity=100000,  # the capacity should be the same as inference.py
        data_shapes=[
            DataShape("observations", (17,)),
            DataShape("actions", (6,)),
            DataShape("rewards", ()),
            DataShape("terminals", (), dtype="bool"),
            DataShape("next_observations", (17,)),
            DataShape("masks", ()),
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
            "masks": LatestSampler(),
        },
    )

    wandb_logger = None

    update_step = 0
    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", "Invalid request type"
        nonlocal wandb_logger, update_step
        if wandb_logger is not None:
            wandb_logger.log(
                {f"env/{k}": v for k, v in payload.items()},
                step=update_step,
            )
        return {}

    # Create server
    server = TrainerServer(make_trainer_config(),
                           log_level=logging.WARNING,
                           request_callback=stats_callback)
    server.register_data_store("train", data_store)

    # Make an agent
    agent = make_agent(
        data_store.dataset["observations"][:1],
        data_store.dataset["actions"][:1],
    )

    server.start(threaded=True)

    # Wait for enough data
    data_pbar = tqdm.trange(10000, desc="Waiting for data")
    while len(data_store) < 10000:
        data_pbar.update(len(data_store) - data_pbar.n)
        time.sleep(1)

    wandb_logger = make_wandb_logger()

    # Training loop
    for train_step in tqdm.tqdm(itertools.count(), desc="Training"):
        # Sample from RB
        batch, mask = data_store.sample("train", 10)

        # Update agent
        agent, update_info = agent.update(batch)
        update_step = train_step

        # Logging
        if train_step % 100 == 0:
            wandb_logger.log(
                update_info,
                step=train_step,
            )

        # Update params
        if train_step % 100 == 0:
            server.publish_network(agent.state.params)


if __name__ == "__main__":
    main()
