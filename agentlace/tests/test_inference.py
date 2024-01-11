#!/usr/bin/env python3

from agentlace.inference import InferenceClient, InferenceServer

def test_inference():
    # Test functions for server interfaces
    def echo(payload: dict) -> dict:
        return {"status": "ok", "message": payload}

    def greet(payload: dict) -> dict:
        name = payload.get("name")
        if name:
            return {"status": "ok", "message": f"Hello, {name}!"}
        else:
            return {"status": "error", "message": "Name not provided"}

    # 1. Set up the server
    port = 12345
    server = InferenceServer(port)
    server.register_interface("echo", echo)
    server.register_interface("greet", greet)

    # Start the server in a separate thread
    server.start(threaded=True)

    # 2. Create a client and test
    client = InferenceClient("localhost", port)
    
    # Test listing interfaces
    available_interfaces = client.interfaces()
    assert "echo" in available_interfaces, "echo interface not found"
    assert "greet" in available_interfaces, "greet interface not found"

    # Test calling echo interface
    response = client.call("echo", {"message": "test"})
    assert response["status"] == "ok", "echo interface failed"
    assert response["message"]["message"] == "test", "echo interface returned incorrect data"

    # Test calling greet interface
    response = client.call("greet", {"name": "John"})
    assert response["status"] == "ok", "greet interface failed"
    assert response["message"] == "Hello, John!", "greet interface returned incorrect data"

    # Test calling greet interface with missing name
    response = client.call("greet", {})
    assert response["status"] == "error", "greet interface failed to handle error"
    assert response["message"] == "Name not provided", "greet interface returned incorrect error"

    # Test calling an unknown interface
    response = client.call("unknown", {})
    assert not response["success"], "Server did not handle unknown interface"

    print("[test_inference] All tests passed!")

    server.stop()
    del server
    del client

if __name__ == "__main__":
    test_inference()
