#!/usr/bin/env python3

from typing import Any, Callable
import zmq
import argparse
import pickle
import logging
import zlib
import threading

##############################################################################

class BroadcastServer:
    def __init__(self, port=5557, log_level=logging.DEBUG):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        logging.basicConfig(level=log_level)
        logging.debug(f"Publisher server is broadcasting on port {port}")

    def broadcast(self, message: dict):
        serialized = pickle.dumps(message)
        serialized = zlib.compress(serialized)
        self.socket.send(serialized)

##############################################################################

class BroadcastClient:
    def __init__(self, ip: str, port=5557, log_level=logging.DEBUG):
        self.context = zmq.Context()
        logging.basicConfig(level=log_level)
        logging.debug(f"Subscriber client is connecting to {ip}:{port}")

        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{ip}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages

        # Set a timeout for the recv method (e.g., 1.5 second)
        self.socket.setsockopt(zmq.RCVTIMEO, 1500)
        self.is_kill = False
        self.thread = None
            
    def async_start(self, callback: Callable[[dict], None]):
        def async_listen():
            while not self.is_kill:
                try:
                    serialized = self.socket.recv()
                    serialized = zlib.decompress(serialized)
                    message = pickle.loads(serialized)
                    callback(message)
                except zmq.Again:
                    # Timeout occurred, check is_kill flag again
                    continue
        threading.Thread(target=async_listen).start()
    
    def stop(self):
        self.is_kill = True # kill the thread
        if self.thread:
            self.thread.join()  # ensure the thread exits
        self.socket.close()

##############################################################################

if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--client', action='store_true')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    args = parser.parse_args()

    if args.server:
        ps = BroadcastServer(port=args.port)
        while True:
            ps.broadcast({'message': "Hello World"})
            time.sleep(1)
    elif args.client:
        pc = BroadcastClient(ip=args.ip, port=args.port)
        pc.async_start(callback=lambda x: print(x))
        print("Listening... asynchonously")
    else:
        raise Exception('Must specify --server or --client')
