#!/usr/bin/env python3

from typing import Any, Optional, Callable, Set, Dict, List
from typing_extensions import Protocol
from collections import deque

from edgeml.zmq_wrapper.req_rep import ReqRepServer, ReqRepClient
from edgeml.zmq_wrapper.broadcast import BroadcastServer, BroadcastClient
from edgeml.internal.utils import compute_hash

import threading
import logging
from pydantic import BaseModel


class TrainerConfig(BaseModel):
    port_number: int
    broadcast_port: int
    queue_size: int = 1


class TrainerCallbackProtocol(Protocol):
    """
    :param payload: Payload to send to the trainer, e.g. training data
    :return: Response to send to the client, can return updated weights.
    """

    def __call__(self, payload: dict) -> Optional[dict]:
        ...

##############################################################################


class TrainerServer:
    def __init__(self,
                 config: TrainerConfig,
                 train_callback: TrainerCallbackProtocol,
                 log_level=logging.DEBUG):
        """
        :param config: Config object
        :param train_callback: Callback function when client requests training
        :param act_callback: Callback function when client requests action
        """
        self.queue = deque()  # FIFO queue

        def __callback_impl(payload: dict) -> dict:
            if len(self.queue) >= config.queue_size:
                self.queue.popleft()
            self.queue.append(payload)
            return train_callback(payload)

        self.req_rep_server = ReqRepServer(config.port_number, __callback_impl)
        self.broadcast_server = BroadcastServer(config.broadcast_port)

    def get_data(self) -> List[dict]:
        """
        Returns the list of training data, max size is `config.queue_size`
        """
        return list(self.queue)

    def publish_weights(self, payload: dict):
        """
        Publishes the weights to the broadcast server,
        This only works if the broadcast server is initialized 
        via `Config.broadcast_port`
        """
        self.broadcast_server.broadcast(payload)

    def start(self, threaded: bool = False):
        """
        Starts the server, defaulting to blocking mode
        :param threaded: Whether to start the server in a separate thread
        """
        if threaded:
            self.thread = threading.Thread(target=self.req_rep_server.run)
            self.thread.start()
        else:
            self.req_rep_server.run()

    def stop(self):
        self.req_rep_server.stop()

##############################################################################


class TrainerClient:
    def __init__(self, server_ip: str, config: TrainerConfig):
        self.req_rep_client = ReqRepClient(server_ip, config.port_number)
        self.server_ip = server_ip
        self.config = config
        print(f"Initiated trainer client at {server_ip}:{config.port_number}")

    def train_step(self, payload: dict) -> Optional[dict]:
        """
        Performs a training step with the given payload.
        NOTE: This is a blocking call. If the trainer needs more time to process,
            use the publish_weights() method in the server instead.
        :param payload: Payload to send to the trainer
        :return: Response from the trainer, return None if timeout
        """
        return self.req_rep_client.send_msg(payload)

    def recv_weights_callback(self, callback: Callable[[dict], None]):
        """Registers the callback function for receiving weights"""
        self.broadcast_client = BroadcastClient(
            self.server_ip, self.config.broadcast_port)
        self.broadcast_client.async_start(callback)

    def stop(self):
        """Stop the client"""
        self.broadcast_client.stop()
