#!/usr/bin/env python3

from agentlace.tests.test_trainer import test_trainer
from agentlace.tests.test_inference import test_inference
from agentlace.tests.test_action import test_action

if __name__ == '__main__':
    test_action()
    test_inference()
    test_trainer()
    print(" [Agentlace] All tests passed!")
