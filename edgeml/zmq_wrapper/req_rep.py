#!/usr/bin/env python3

import zmq
import argparse
from typing import Optional, Callable, Dict
from typing_extensions import Protocol
import pickle
import logging
import zlib

##############################################################################

class CallbackProtocol(Protocol):
    def __call__(self, message: Dict) -> Dict:
        ...

class ReqRepServer:
    def __init__(self,
                 port=5556,
                 impl_callback: Optional[CallbackProtocol]=None,
                 log_level=logging.DEBUG):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.socket.setsockopt(zmq.SNDHWM, 5)
        self.impl_callback = impl_callback

        # Set a timeout for the recv method (e.g., 1.5 second)
        self.socket.setsockopt(zmq.RCVTIMEO, 1500)
        self.is_kill = False
        self.thread = None

        logging.basicConfig(level=log_level)
        logging.debug(f"Req-rep server is listening on port {port}")

    def run(self):
        while not self.is_kill:
            try:
                #  Wait for next request from client
                message = self.socket.recv()
                message = zlib.decompress(message)
                message = pickle.loads(message)
                logging.debug(f"Received new request: {message}")

                #  Send reply back to client
                if self.impl_callback:
                    res = self.impl_callback(message)
                    res = pickle.dumps(res)
                    res = zlib.compress(res)
                    self.socket.send(res)
                else:
                    logging.warning("No implementation callback provided.")
                    self.socket.send(b"World")
            except zmq.Again as e:
                continue

    def stop(self):
        self.is_kill = True # kill the thread in run
        if self.thread:
            self.thread.join()  # ensure the thread exits
        self.socket.close()

##############################################################################

class ReqRepClient:
    def __init__(self,
                 ip: str,
                 port=5556,
                 timeout_ms = 800,
                 log_level=logging.DEBUG):
        """
        :param ip: IP address of the server
        :param port: Port number of the server
        :param timeout_ms: Timeout in milliseconds
        :param log_level: Logging level, defaults to DEBUG
        """
        self.context = zmq.Context()
        logging.basicConfig(level=log_level)
        logging.debug(f"Req-rep client is connecting to {ip}:{port}")

        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{ip}:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

    def send_msg(self, request: dict) -> Optional[str]:
        # pickle is chosen over protobuf due to faster de/serialization process
        # https://medium.com/@shmulikamar/python-serialization-benchmarks-8e5bb700530b
        serialized = pickle.dumps(request)
        serialized = zlib.compress(serialized)
        try:
            self.socket.send(serialized)
            message = self.socket.recv()
            message = zlib.decompress(message)
            return pickle.loads(message)
        except Exception as e:
            # accepts timeout exception
            return None


##############################################################################

if __name__ == "__main__":
    # NOTE: This is just for Testing
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--client', action='store_true')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    args = parser.parse_args()

    def do_something(message):
        return b'World'

    if args.server:
        ss = ReqRepServer(port=args.port, impl_callback=do_something)
        ss.run()
    elif args.client:
        sc = ReqRepClient(ip=args.ip, port=args.port)
        r = sc.send_msg({'hello': 1})
        print(r)
    else:
        raise Exception('Must specify --server or --client')
