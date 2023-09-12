# edgeml

It is common for edge device to be limited by GPU compute. This library enables distributed datastream from edge device to GPU compute for various ml applications. The lib mainly based on client-server architecutre, enable simple TCP communication between multiple clients to server.

> THIS IS A WORK IN PROGRESS, The code is still in testing phase.

## Installation

```bash
pip install -e .
```

## Main classes:

1. Edge device as server: `edgeml.EdgeServer` and `edgeml.EdgeClient`
   - `EdgeServer` provides observation to client
   - `EdgeClient` can provide further action to server (Optional)

2. Trainer compute as server: `edgeml.TrainerServer` and `edgeml.TrainerClient`
   - `TrainerClient` provides observation to server and gets new weights

3. Inference compute as server: `edgeml.InferenceServer` and `edgeml.InferenceClient`
   - `InferenceClient` provides observation to server and gets prediction

## Usage

1. **Edge Device as Server**
An edge device (Agent) can send observations to a remote client. The client, in turn, can provide actions to the agent based on these observations.

This uses the `edgeml.EdgeServer` and `edgeml.EdgeClient` classes.


Inference compute as client
```py
import edgeml

model = load_model()
agent = edgeml.EdgeClient('localhost', 6379, task_id='mnist', config=agent_config)

for _ in range(100):
    observation = agent.get_observation()
    prediction = model.predict(observation)
    agent.send_action(prediction)
```

Edge device as server
```py
def action_callback(key, action):
    # TODO: process action here
    return {"status": "received"}

def observation_callback(keys):
    # TODO: return the desired observations here
    return {"observation": "some_value"}

config = edgeml.EdgeConfig(port_number=6379, action_keys=['action'], observation_keys=['observation'])
agent_server = edgeml.EdgeServer(config, observation_callback, action_callback)
agent_server.start()
```

2. **Remote Training Example for an RL Application**
A remote trainer receives observations from an edge device (Agent) and sends updated weights back. The Agent then updates its model with these new weights.

This uses the `edgeml.TrainerServer` and `edgeml.TrainerClient` classes.

Client

```py
def main():
    env = gym.make('CartPole-v0')
    observation = env.reset()
    config = edgeml.TrainerConfig(port_number=6379, broadcast_port=6380, payload_keys={"observation"})
    trainer = edgeml.TrainerClient('localhost', config)

    agent = make_agent()  # Arbitrary agent

    while True:
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)

        # or we can use callback function to receive new weights
        new_weights = trainer.train_step({"observation": observation})

        agent.update_weights(new_weights)
```

Trainer (Remote compute)

```py
def train_step(payload):
    observation = payload["observation"]
    # TODO: do some training based on observation
    new_weights = {}  # Create new weights here
    return new_weights

def main():
    config = edgeml.TrainerConfig(port_number=6379, broadcast_port=6380, payload_keys={"observation"})
    trainer_server = edgeml.TrainerServer(config, train_step)
    trainer_server.start()
```

3. **Agent as client and inference as server**

This uses the `edgeml.InferenceServer` and `edgeml.InferenceClient` classes. This is useful for low power edge devices that cannot run inference locally.

Inference server
```py
import edgeml

def predict(payload):
    # TODO: do some prediction based on payload
    return {"prediction": "some_value"}

inference_server = edgeml.InferenceServer(port_num=6379)
inference_server.register_interface("voice_reg", predict)
```

client inference
```py
import edgeml

client = edgeml.InferenceClient('localhost', 6379)
res = client.call("voice_reg", {"audio": "serialized_audio"})
```
