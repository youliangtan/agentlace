#!/usr/bin/env python

import gym
import logging
import time
from agentlace.action import ActionClient, ActionServer, ActionConfig

logging.basicConfig(level=logging.INFO)

# These describes the ports and API keys used in the agentlace server
DefaultActionConfig = ActionConfig(
    port_number=5556,
    action_keys=["reset", "step", "render", "close"],
    observation_keys=["observation_space", "action_space"],
    broadcast_port=5556 + 1,
)


class ActionClientEnv(gym.Env):
    """Action Client interface with agentlace action server"""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 5556,
                 retry_interval: float = 0.5,
                 ):
        """
        Initialize the action client interface.
        Args:
            host: the host of the server
            port: the port number of the server
            retry_interval: the interval to retry if no response
        """
        _config = DefaultActionConfig
        _config.port_number = port
        _config.broadcast_port = port + 1
        self.retry_interval = retry_interval
        self._client = ActionClient(host, _config)

        # create action client
        _full_obs = self._try_obs()
        self.observation_space = _full_obs["observation_space"]
        self.action_space = _full_obs["action_space"]

    def step(self, action):
        """Standard gym step function."""
        return self._try_act("step", action)

    def reset(self, **kwargs):
        """Standard gym reset function."""
        return self._try_act("reset", kwargs)

    def render(self, **kwargs):
        """Standard gym render function."""
        return self._try_act("render", kwargs)

    def close(self):
        """Standard gym close function."""
        return self._try_act("close", {})

    def _try_obs(self):
        obs = self._client.obs()
        while obs is None:
            print("waiting for observation")
            obs = self._client.obs()
            time.sleep(self.retry_interval)
        return obs

    def _try_act(self, act_key, act_payload):
        res = self._client.act(act_key, act_payload)
        while res is None:
            print(f"waiting for action {act_key}")
            res = self._client.act(act_key, act_payload)
            time.sleep(self.retry_interval)
        return res["act_ret"]


class ActionServerEnvWrapper(gym.Wrapper):
    """
    This wraps the gym environment to provide the server interface.
    """

    def __init__(self,
                 env: gym.Env,
                 port: int = 5556,
                 ):
        """
        Provide the manipulator interface to the server.
        args:
            env: the gym environment to wrap
            port: the port number to run the server
        """
        super().__init__(env)
        _config = DefaultActionConfig
        _config.port_number = port
        _config.broadcast_port = port + 1

        self.__server = ActionServer(
            _config,
            obs_callback=self.__observe,
            act_callback=self.__action,
            # hide all logs
            log_level=logging.INFO,
        )
        print(f"initializing env action server with port: {port}")

    def start(self, threaded: bool = False):
        """
        This starts the server. Default is blocking.
        """
        self.__server.start(threaded)

    def stop(self):
        """Stop the server."""
        self.__server.stop()

    def __observe(self, types: list) -> dict:
        obs = {
            "action_space": self.env.action_space,
            "observation_space": self.env.observation_space,
        }
        return obs

    def __action(self, type: str, req_payload) -> dict:
        if type == "reset":
            ret_tuple = self.env.reset(**req_payload)
        elif type == "step":
            ret_tuple = self.env.step(req_payload)
        elif type == "render":
            ret_tuple = self.env.render(**req_payload)
        elif type == "close":
            ret_tuple = self.env.close()
        else:
            raise ValueError(f"Invalid action type: {type}")
        return {"act_ret": ret_tuple}
