#!/usr/bin/env python3

from typing import Optional, Callable, Set, Dict, Protocol, List
from edgeml.internal.client import Client
from edgeml.internal.server import Server
from edgeml.zmq_wrapper.req_rep import ReqRepServer, ReqRepClient
from edgeml.internal.utils import compute_hash
import threading
from pydantic import BaseModel

##############################################################################

class EdgeConfig(BaseModel):
    port_number: int
    action_keys: List[str]
    observation_keys: List[str]

class CallbackProtocol(Protocol):
    """
    Define the data type of the callback function
    :param keys: Set of keys requested by the client, defaults to all keys.
                 User-side filtering is recommended.
    :param payload: Payload sent by the client, defaults to empty dict
    :return: Response to send to the client
    """
    def __call__(self, keys: Set, payload: Dict) -> Dict:
        ...

class EdgeServer:
    def __init__(self,
                 config: EdgeConfig,
                 obs_callback: Optional[CallbackProtocol],
                 act_callback: Optional[CallbackProtocol]):
        """
        :param config: Config object
        :param obs_callback: Callback function when client requests observation
        :param act_callback: Callback function when client requests action
        """
        def obs_impl(payload: dict) -> dict:
            if payload["type"] == "obs" and obs_callback is not None:
                return obs_callback(keys=payload["keys"], payload=payload["payload"])
            elif payload["type"] == "act" and act_callback is not None:
                return act_callback(key=payload["key"], payload=payload["payload"])
            elif payload["type"] == "hash":
                print(config.json())
                return {"payload": compute_hash(config.json())}
            return {"status": "error", "message": "Invalid payload"}

        self.server = ReqRepServer(config.port_number, obs_impl)

    def create_obs_streamer(self):
        raise NotImplementedError

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

class EdgeClient:
    def __init__(self, server_ip: str, config: EdgeConfig):
        self.client = ReqRepClient(server_ip, config.port_number)
        # Check hash of server config to ensure compatibility
        res = self.client.send_msg({"type": "hash"})
        if res is None:
            raise Exception("Failed to connect to server")
        local_hash = compute_hash(config.json())
        if local_hash != res["payload"]:
            raise Exception(
                f"Incompatible config with hash with server. "
                "Please check the config of the server and client")

        # use hash for faster lookup. config uses list because it is
        # used for hash md4 comparison
        config.observation_keys = set(config.observation_keys)
        config.action_keys = set(config.action_keys)
        self.config = config
        print("Done init client")

    def obs(self, keys: Optional[Set[str]]=None, payload={}) -> dict:
        """
        Returns the observation from the server
        :param keys: Set of keys to request from the server, defaults to all keys
        :param payload: Payload to send to the server
        :return: Observation from the server
        """
        if keys is not None:
            for key in keys:
                assert key in self.config.observation_keys,\
                    f"Invalid observation key: {key}"
        messsage = {"type": "obs", "keys": keys, "payload": payload}
        return self.client.send_msg(messsage)

    def act(self, key:str, payload: dict={}) -> dict:
        """
        Sends the action to the server
        :param key: Key of the action
        :param payload: Payload to send to the server
        :return: Response from the server
        """
        if key not in self.config.action_keys:
            raise Exception(f"Invalid action key: {key}")
        message = {"type": "act", "key": key, "payload": payload}
        return self.client.send_msg(message)

    def register_obs_callback(self, callback: Callable):
        """Registers the callback function for observation streaming"""
        raise NotImplementedError

##############################################################################

class InterferenceServer:
    def __init__(self, port_num: int):
        pass
    
    def start(threaded: bool = False):
        """
        Starts the server, defaulting to blocking mode
        :param threaded: Whether to start the server in a separate thread
        """
        pass
    
    def register_interface(self, name: str, callback: Callable):
        """
        Registers the callback function for the interface
        :param name: Name of the interface
        :param callback: Callback function for the interface
        """
        pass

class InterferenceClient:
    def __init__(self, server_ip: str, port_num: int):
        pass
    
    def interfaces(self) -> Set[str]:
        """
        Returns the set of interfaces available on the server
        :return: Set of interfaces available on the server
        """
        return set()

    def call(self, name: str, payload: dict) -> Optional[dict]:
        """
        Calls the interface on the server with the given payload
        :param name: Name of the interface
        :param payload: Payload to send to the interface
        :return: Response from the interface
        """
        return {}

##############################################################################

class TrainerConfig:
    port_number: int
    payload_keys: Set[str]
    response_format: str

class TrainerServer:
    def __init__(self, config: TrainerConfig):
        pass

    def train_step(self, payload: dict) -> Optional[dict]:
        """
        Performs a training step with the given payload
        :param payload: Payload to send to the trainer
        :return: Response from the trainer
        """
        raise NotImplementedError

    def start(threaded: bool = False):
        """
        Starts the server, defaulting to blocking mode
        :param threaded: Whether to start the server in a separate thread
        """
        pass

class TrainerClient:
    def __init__(self, server_ip: str, config: TrainerConfig):
        pass

    def train_step(self, payload: dict) -> Optional[dict]:
        """
        Performs a training step with the given payload
        :param payload: Payload to send to the trainer
        :return: Response from the trainer
        """
        return None
