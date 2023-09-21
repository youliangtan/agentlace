#!/usr/bin/env python3

from edgeml.tests.test_trainer import test_trainer
from edgeml.tests.test_inference import test_inference
from edgeml.tests.test_action import test_action

if __name__ == '__main__':
    test_action()
    test_inference()
    test_trainer()
    print(" [EdgeML] All tests passed!")
