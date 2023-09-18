#!/usr/bin/env python3

from typing import Optional, Callable, Set, Dict, List
from typing_extensions import Protocol
from collections import deque

from edgeml.zmq_wrapper.req_rep import ReqRepServer, ReqRepClient
from edgeml.zmq_wrapper.broadcast import BroadcastServer, BroadcastClient
from edgeml.internal.utils import compute_hash

import time
import threading
import logging
from pydantic import BaseModel
import json

##############################################################################


class ActorConfig(BaseModel):
    """Configuration for the edge server and client
    NOTE: Client and server should have the same config
            client will raise Error if configs are different, thus can use
            `version` to check for compatibility
    """
    port_number: int
    action_keys: List[str]
    observation_keys: List[str]
    broadcast_port: Optional[int] = None
    version: str = "0.0.1"


class ObsCallback(Protocol):
    """
    Define the data type of the callback function
        :param keys: Set of keys of the observation, defaults to all keys
        :return: Response to send to the client
    """

    def __call__(self, keys: Set) -> Dict:
        ...


class ActCallback(Protocol):
    """
    Define the data type of the callback function
        :param key: Key of the action
        :param payload: Payload related to the action in dict format
        :return: Response to send to the client
    """

    def __call__(self, key: str, payload: Dict) -> Dict:
        ...

##############################################################################


class ActorServer:
    def __init__(self,
                 config: ActorConfig,
                 obs_callback: Optional[ObsCallback],
                 act_callback: Optional[ActCallback],
                 log_level=logging.DEBUG):
        """
        Args:
            :param config: Config object
            :param obs_callback: Callback function when client requests observation
            :param act_callback: Callback function when client requests action
        """
        def __obs_parser_cb(payload: dict) -> dict:
            if payload["type"] == "obs" and obs_callback is not None:
                return obs_callback(payload["keys"])
            elif payload["type"] == "act" and act_callback is not None:
                return act_callback(payload["key"], payload["payload"])
            elif payload["type"] == "hash":
                config_json = json.dumps(config.dict(), separators=(',', ':'))
                return {"status": "success", "payload": config_json}
            return {"status": "error", "message": "Invalid payload"}

        self.config = config
        self.server = ReqRepServer(config.port_number, __obs_parser_cb)
        logging.basicConfig(level=log_level)
        logging.debug(f"Edge server is listening on port {config.port_number}")

        if config.broadcast_port is not None:
            self.broadcast = BroadcastServer(config.broadcast_port)
            logging.debug(f"Broadcast server is on port {config.broadcast_port}")

    def start(self, threaded: bool = False):
        """
        Starts the server, defaulting to blocking mode
            :param threaded: Whether to start the server in a separate thread
        """
        if threaded:
            self.thread = threading.Thread(target=self.server.run)
            self.thread.start()
        else:
            self.server.run()

    def publish_obs(self, payload: dict) -> bool:
        """
        Publishes the observation to the broadcast server,
        Enable broadcasting by defining `broadcast_port` in `ActorConfig`
        """
        if self.config.broadcast_port is None:
            logging.warning("Broadcast server not initialized")
            return False
        self.broadcast.broadcast(payload)
        return True

    def stop(self):
        """Stop the server"""
        self.server.stop()

##############################################################################


class ActorClient:
    def __init__(self, server_ip: str, config: ActorConfig):
        self.client = ReqRepClient(server_ip, config.port_number)
        # Check hash of server config to ensure compatibility
        res = self.client.send_msg({"type": "hash"})
        if res is None:
            raise Exception("Failed to connect to server")

        config_json = json.dumps(config.dict(), separators=(',', ':'))
        if compute_hash(config_json) != compute_hash(res["payload"]):
            raise Exception(
                f"Incompatible config with hash with server. "
                "Please check the config of the server and client")

        # use hash for faster lookup. config uses list because it is
        # used for hash md5 comparison
        config.observation_keys = set(config.observation_keys)
        config.action_keys = set(config.action_keys)
        self.config = config
        self.server_ip = server_ip
        self.broadcast_client = None

    def obs(self, keys: Optional[Set[str]] = None) -> Optional[dict]:
        """
        Get the observation from the Edge Server
            :param keys: Set of keys to request, `None`: select all
            :return: Observation from the server, `None` when not connected
        """
        if keys is None:
            keys = self.config.observation_keys  # if None and select all
        else:
            for key in keys:
                assert key in self.config.observation_keys, f"Invalid obs key: {key}"
        messsage = {"type": "obs", "keys": keys}
        return self.client.send_msg(messsage)

    def act(self, key: str, payload: dict = {}) -> Optional[dict]:
        """
        Sends the action to the server
            :param key: Key of the action
            :param payload: Payload to send to the server
            :return: Response from the server, `None` when not connected
        """
        if key not in self.config.action_keys:
            raise Exception(f"Invalid action key: {key}")
        message = {"type": "act", "key": key, "payload": payload}
        return self.client.send_msg(message)

    def register_obs_callback(self, callback: Callable):
        """
        Registers the callback function for observation streaming
            :param callback: Callback function for observation streaming
        """
        if self.config.broadcast_port is None:
            raise Exception("Broadcast server not initialized")
        self.broadcast_client = BroadcastClient(
            self.server_ip, self.config.broadcast_port)
        self.broadcast_client.async_start(callback)

    def stop(self):
        """Stop the client"""
        if self.broadcast_client is not None:
            self.broadcast_client.stop()

##############################################################################


class InferenceServer:
    def __init__(self, port_num: int):
        """
        Create a server with the given port number and interface names
        """
        def __parser_cb(payload: dict) -> dict:
            # if is list_interfaces type
            if payload.get('type') == 'list_interfaces':
                return {"interfaces": list(self.interfaces.keys())}
            elif payload.get('type') == 'call_interface':
                interface_name = payload.get('interface')
                if interface_name and interface_name in self.interfaces:
                    return self.interfaces[interface_name](payload['payload'])
            return {"status": "error", "message": "Invalid interface or payload"}

        self.server = ReqRepServer(port_num, __parser_cb)
        self.interfaces = {}

    def start(self, threaded: bool = False):
        """
        Starts the server, defaulting to blocking mode
            :param threaded: Whether to start the server in a separate thread
        """
        if threaded:
            self.thread = threading.Thread(target=self.server.run)
            self.thread.start()
        else:
            self.server.run()

    def register_interface(self, name: str, callback: Callable):
        """
        Registers the callback function for the interface
            :param name: Name of the interface
            :param callback: Callback function for the interface
        """
        self.interfaces[name] = callback

    def stop(self):
        """Stop the server"""
        self.server.stop()

##############################################################################


class InferenceClient:
    def __init__(self, server_ip: str, port_num: int):
        self.client = ReqRepClient(server_ip, port_num)

    def interfaces(self) -> Set[str]:
        """
        Returns the set of interfaces available on the server
            :return: Set of interfaces available on the server
        """
        response = self.client.send_msg({"type": "list_interfaces"})
        if response:
            return set(response.get('interfaces', []))
        return set()

    def call(self, name: str, payload: dict) -> Optional[dict]:
        """
        Calls the interface on the server with the given payload
            :param name: Name of the interface
            :param payload: Payload to send to the interface
            :return: Response from the interface
        """
        return self.client.send_msg(
            {"type": "call_interface", "interface": name, "payload": payload})

##############################################################################


class TrainerConfig(BaseModel):
    """
    Configuration for the edge server and client
    NOTE: Client and server should have the same config
          client will raise Error if configs are different, thus can use
          `version` to check for compatibility
    """
    port_number: int
    broadcast_port: int
    queue_size: int = 1
    request_types: List[str] = []
    rate_limit: Optional[int] = None
    version: str = "0.0.1"


class TrainerCallback(Protocol):
    """
    Define the data type of the trainer callback function
        :param payload: Payload to send to the trainer, e.g. training data
        :return: Response to send to the client, can return updated weights.
    """

    def __call__(self, payload: dict) -> dict:
        ...


class RequestCallback(Protocol):
    """
    Define the data type of the request callback function
        :param type: Name of the custom request
        :param payload: Payload to send to the trainer, e.g. training data
        :return: Response to send to the client, can return updated weights.
    """

    def __call__(self, type: str, payload: dict) -> dict:
        ...


##############################################################################


class TrainerServer:
    def __init__(self,
                 config: TrainerConfig,
                 train_callback: TrainerCallback,
                 request_callback: Optional[RequestCallback] = None,
                 log_level=logging.DEBUG):
        """
        Args
            :param config: Config object
            :param train_callback: Callback fn when client call `train_step`
            :param request_callback: Callback fn to impl custom requests
        """
        self.queue = deque()  # FIFO queue
        self.request_types = set(config.request_types)  # faster lookup

        def __callback_impl(data: dict) -> dict:
            _type, _payload = data.get("type"), data.get("payload")
            if _type == "data":
                if len(self.queue) >= config.queue_size:
                    self.queue.popleft()
                self.queue.append(_payload)
                return train_callback(_payload)
            elif _type == "hash":
                print("hash", config.json())
                config_json = json.dumps(config.dict(), separators=(',', ':'))
                return {"status": "success", "payload": config_json}
            elif _type in self.request_types:
                return request_callback(_type, _payload) if request_callback else {}
            return {"status": "error", "message": "Invalid type or payload"}

        self.req_rep_server = ReqRepServer(config.port_number, __callback_impl)
        self.broadcast_server = BroadcastServer(config.broadcast_port)
        logging.basicConfig(level=log_level)
        logging.debug(f"Trainer server is listening on port {config.port_number}")

    def get_data(self) -> List[dict]:
        """
        Get the list of training data, max size is `config.queue_size`
        """
        return list(self.queue)

    def publish_weights(self, payload: dict):
        """
        Publishes the weights to the broadcast server.
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
                 server_ip: str,
                 config: TrainerConfig,
                 log_level=logging.DEBUG):
        """
        Args:
            :param server_ip: IP address of the server
            :param config: Config object
            :param log_level: Logging level
        """
        self.req_rep_client = ReqRepClient(server_ip, config.port_number)
        self.server_ip = server_ip
        self.config = config
        self.request_types = set(config.request_types)  # faster lookup

        res = self.req_rep_client.send_msg({"type": "hash"})
        if res is None:
            raise Exception("Failed to connect to server")
        config_json = json.dumps(config.dict(), separators=(',', ':'))
        if compute_hash(config_json) != compute_hash(res["payload"]):
            raise Exception(
                f"Incompatible config with hash with server. "
                "Please check the config of the server and client")
        self.last_request_time = 0
        logging.basicConfig(level=log_level)
        logging.debug(
            f"Initiated trainer client at {server_ip}:{config.port_number}")

    def train_step(self, data: dict) -> Optional[dict]:
        """
        Performs a training step with the given payload.
        NOTE: This is a blocking call. If the trainer needs more time to process,
        use the publish_weights() method in the server instead.
            :param data: Payload to send to the trainer
            :return: Response from the trainer, return None if timeout
        """
        msg = {"type": "data", "payload": data}
        if self.config.rate_limit and \
                time.time() - self.last_request_time < 1 / self.config.rate_limit:
            logging.warning("Rate limit exceeded")
            return None
        self.last_request_time = time.time()
        return self.req_rep_client.send_msg(msg)

    def request(self, type: str, payload: dict) -> Optional[dict]:
        """
        Call the trainer server with custom requests
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

    def recv_weights_callback(self, callback: Callable[[dict], None]):
        """Registers the callback function for receiving weights"""
        self.broadcast_client = BroadcastClient(
            self.server_ip, self.config.broadcast_port)
        self.broadcast_client.async_start(callback)

    def stop(self):
        """Stop the client"""
        self.broadcast_client.stop()
        logging.debug("Stopped trainer client")
