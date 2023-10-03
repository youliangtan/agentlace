#!/usr/bin/env python3

from __future__ import annotations
from typing import Any, Optional, Callable, Set, Dict, List, Tuple, Any
from typing_extensions import Protocol
from collections import deque

from edgeml.zmq_wrapper.req_rep import ReqRepServer, ReqRepClient
from edgeml.zmq_wrapper.broadcast import BroadcastServer, BroadcastClient
from edgeml.internal.utils import compute_hash
from edgeml.data.data_store import DataStoreBase

import time
import threading
import logging
from pydantic import BaseModel
import json


##############################################################################


class TrainerConfig(BaseModel):
    """
    Configuration for the edge server and client
    NOTE: Client and server should have the same config
          client will raise Error if configs are different, thus can use
          `version` to check for compatibility
    """
    port_number: int = 5555
    broadcast_port: int = 5556
    request_types: List[str] = []
    rate_limit: Optional[int] = None
    version: str = "0.0.1"


class DataCallback(Protocol):
    """
    Define the data type of the data callback function
        :param store_name: Name of the data store
        :param payload: Payload to send to the trainer, e.g. training data
    """

    def __call__(self, store_name, payload: dict):
        ...


class RequestCallback(Protocol):
    """
    Define the data type of the request callback function
        :param type: Name of the custom request
        :param payload: Payload to send to the trainer, e.g. training data
        :return: Response to send to the client, can return updated network.
    """

    def __call__(self, type: str, payload: dict) -> dict:
        ...


##############################################################################


class TrainerServer:
    def __init__(self,
                 config: TrainerConfig,
                 data_callback: Optional[DataCallback] = None,
                 request_callback: Optional[RequestCallback] = None,
                 log_level=logging.INFO):
        """
        Args
            :param config: Config object
            :param data_callback: Callback fn when a new data arrives
            :param request_callback: Callback fn to impl custom requests
        """
        self.queue = deque()  # FIFO queue
        self.request_types = set(config.request_types)  # faster lookup
        self.data_stores = {}
        self.config = config

        def __callback_impl(data: dict) -> dict:
            _type, _payload = data.get("type"), data.get("payload")
            if _type == "data":
                store_name = data.get("store_name")
                if store_name not in self.data_stores:
                    return {"success": False, "message": "Invalid table name"}
                indices, data = _payload.get("indices", []), _payload.get("data", {})
                self.data_stores[store_name].update_data(indices, data)
                if data_callback:
                    data_callback(store_name, _payload)
                return {"success": True}
            elif _type == "hash":
                config_json = json.dumps(config.dict(), separators=(',', ':'))
                return {"success": True, "payload": config_json}
            elif _type in self.request_types:
                return request_callback(_type, _payload) if request_callback else {}
            return {"success": False, "message": "Invalid type or payload"}

        self.req_rep_server = ReqRepServer(config.port_number, __callback_impl, log_level=log_level)
        self.broadcast_server = BroadcastServer(config.broadcast_port, log_level=log_level)
        logging.basicConfig(level=log_level)
        logging.debug(f"Trainer server is listening on port {config.port_number}")

    def register_data_store(self, name, data_store: DataStoreBase):
        """Register a datastore to the server"""
        self.data_stores[name] = data_store

    def data_store(self, name) -> Optional[DataStoreBase]:
        """Get the datastore reference from the server"""
        return self.data_stores.get(name)

    def store_names(self) -> Set[str]:
        """
        Returns the set of table names available on the server
            :return: Set of table names available on the server
        """
        return set(self.data_stores.keys())

    def publish_network(self, payload: dict):
        """
        Publishes the network to the broadcast server.
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
        """Stop the server"""
        self.req_rep_server.stop()


##############################################################################


class TrainerClient:
    def __init__(self,
                 name: str,
                 server_ip: str,
                 config: TrainerConfig,
                 data_store: DataStoreBase,
                 log_level=logging.INFO):
        """
        Args:
            :param name: Name of the client, creates a unique datastore
            :param server_ip: IP address of the server
            :param config: Config object
            :param data_store: Datastore to store the data
            :param log_level: Logging level
        """
        self.name = name
        self.req_rep_client = ReqRepClient(server_ip, config.port_number, log_level=log_level)
        self.server_ip = server_ip
        self.config = config
        self.request_types = set(config.request_types)  # faster lookup
        self.data_store = data_store
        self.last_sync_data_id = -1
        self.update_thread = None
        self.last_request_time = 0
        self.log_level = log_level
        logging.basicConfig(level=log_level)

        res = self.req_rep_client.send_msg({"type": "hash"})
        if res is None:
            raise Exception("Failed to connect to server")

        # try check configuration compatibility
        config_json = json.dumps(config.dict(), separators=(',', ':'))
        if compute_hash(config_json) != compute_hash(res["payload"]):
            raise Exception(
                f"Incompatible config with hash with server. "
                "Please check the config of the server and client")

        # First update the server's datastore
        res = self.update()
        if res is None or not res["success"]:
            logging.error("Failed to update server's data store, do check!")
        logging.debug(
            f"Initiated trainer client at {server_ip}:{config.port_number}")

    def update(self) -> Optional[dict]:
        """
        This will explicity trigger an update to the data store
            :return Response from the trainer, return None if timeout
        """
        latest_id = self.data_store.latest_data_id()
        indices, data = self.data_store.get_latest_data(self.last_sync_data_id)
        res = self._update({"indices": indices, "data": data})
        if res and res["success"]:
            self.last_sync_data_id = latest_id
        else:
            logging.warning("Failed to update data")
        return res

    def _update(self, data: dict) -> Optional[dict]:
        """
        internal update method that will send the data to the server
            :param data: Payload to send to the trainer
            :return: Response from the trainer, return None if timeout
        """
        msg = {"type": "data", "store_name": self.name, "payload": data}
        if self.config.rate_limit and \
                time.time() - self.last_request_time < 1 / self.config.rate_limit:
            logging.warning("Rate limit exceeded")
            return None
        self.last_request_time = time.time()
        return self.req_rep_client.send_msg(msg)

    def request(self, type: str, payload: dict) -> Optional[dict]:
        """
        Call the trainer server with custom requests
        Some common use cases: checkpointing, get-stats, etc.
            :param type: Name of the custom request, defined in `config.request_types`
            :param payload: Payload to send to the trainer
            :return: Response from the trainer, return None if timeout
        """
        if type not in self.request_types:
            return None
        msg = {"type": type, "payload": payload}
        if self.config.rate_limit and \
                time.time() - self.last_request_time < 1 / self.config.rate_limit:
            logging.warning("Rate limit exceeded")
            return None
        self.last_request_time = time.time()
        return self.req_rep_client.send_msg(msg)

    def recv_network_callback(self, callback: Callable[[dict], None]):
        """Registers the callback function for receiving an updated network"""
        self.broadcast_client = BroadcastClient(
            self.server_ip, self.config.broadcast_port, log_level=self.log_level)
        self.broadcast_client.async_start(callback)

    def start_async_update(self, interval: int = 10):
        """
        Start a thread that server's datastore every `interval` seconds.
        :param interval: Interval in seconds.
        """
        def _periodic_update():
            while not self.stop_update_flag.is_set():
                self.update()
                time.sleep(interval)

        self.stop_update_flag = threading.Event()
        if self.update_thread is None or not self.update_thread.is_alive():
            self.stop_update_flag.clear()
            self.update_thread = threading.Thread(target=_periodic_update)
            self.update_thread.start()

    def stop(self):
        """Stop the client"""
        self.broadcast_client.stop()
        if self.update_thread and self.update_thread.is_alive():
            self.stop_update_flag.set()
            self.update_thread.join()
        logging.debug("Stopped trainer client")
