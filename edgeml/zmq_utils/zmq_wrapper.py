#!/usr/bin/env python3

import zmq
import argparse
from typing import Optional, Callable

##############################################################################

class SocketServer:
    def __init__(self, port=5556, impl_callback=Callable):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.socket.setsockopt(zmq.SNDHWM, 5)
        self.impl_callback = impl_callback
        print("Socket server is listening...")

    def run(self):
        while True:
            try:
                #  Wait for next request from client
                message = self.socket.recv(flags=zmq.NOBLOCK)
                print(f"Received request: {message}")
                #  Send reply back to client

                if self.impl_callback:
                    res = self.impl_callback(message)
                    self.socket.send(res)
                else:
                    print("WARNING: No implementation callback provided.")
                    self.socket.send(b"World")
            except zmq.Again as e:
                pass

##############################################################################

class SocketClient:
    def __init__(self, ip, port=5556, timeout_ms = 800):
        self.context = zmq.Context()
        #  Socket to talk to server
        print("Connecting to socket server…")
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{ip}:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

    def send_payload(self, request) -> Optional[str]:
        # print(f"Sending request {request} …")
        encoded_str = request.encode()
        try:
            self.socket.send(encoded_str)
            message = self.socket.recv()
            return message
            # print(f"Received reply {request} [ {message} ]")
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

    if args.server:
        ss = SocketServer(port=args.port)
        ss.run()
    elif args.client:
        sc = SocketClient(ip=args.ip, port=args.port)
        r = sc.send_payload('hello')
        print(r)
    else:
        raise Exception('Must specify --server or --client')
