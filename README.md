# edgeml
Library to enable distributed datastream of inference/trainer computes and edge agents for ml applications.

> THIS IS A WORK IN PROGRESS, only skeleton code is provided. The code is not tested and is not ready for use.

## Installation

```bash
pip install edgeml
```

## Main classes:

1. Edge device as server: `edgeml.EdgeServer` and `edgeml.EdgeClient`
   - `EdgeServer` provides observation to client
   - `EdgeClient` can provide further action to server (Optional)

2. Inference compute as server: `edgeml.InferenceServer` and `edgeml.InferenceClient`
   - `InferenceClient` provides observation to server and gets prediction

3. Trainer compute as server: `edgeml.TrainerServer` and `edgeml.TrainerClient`
   - `TrainerClient` provides observation to server and gets new weights

## Usage

1. Remote Training example for an RL application. A edge device (Agent) will provide observation to the trainer, and trainer will return the new weights to the agent. The agent will then update its model with the new weights. 


Agent (edge client)

```py
import edgeml
import torch
import gym

def main():
    env = gym.make('CartPole-v0')
    observation = env.reset()

    trainer = edgeml.TrainerClient('localhost', 6379, task_id='cartpole-v0')
    agent = make_agent()

    while True:
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)

        trainer.send_observation(observation)
        new_weights = trainer.get_weights()

        agent.update_weights(new_weights)
```

Trainer (Remote compute)

```py
import edgeml
import torch
import gym

def train_step(observation):
    # TODO: do some training
    return new_weights

def main():
    env = gym.make('CartPole-v0')
    observation = env.reset()
    trainer_server = edgeml.TrainerServer('localhost', 6379, task_id='cartpole-v0')
    trainer_server.set_weights(make_weights())
    trainer_server.register_train_step(train_step)
    trainer_server.start()
```

2. Remote "control" of edge device. An edge device (Agent) can provide observation to a remote client. The client can then provide an action to the agent. The observation and action spaces are defined during the initialization of the agent and client. An example for RL is provided


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

edge as server
```py
import edgeml

def action_callback(action):
    # some action
    return

def observation_callback(observation):
    # some action
    return

agent_server = edgeml.EdgeServer('localhost', 6379, task_id='mnist')
agent_server.register_handler(observation_callback, action_callback)
agent_server.start()
```

3. Agent as client and inference as server


Client on the edge
```py
import edgeml

inference_compute = edgeml.InferenceClient('localhost', 6379, task_id='mnist')
```

## Notes
