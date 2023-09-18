#!/usr/bin/env python3

from edgeml.tests.test_trainer import test_trainer
from edgeml.tests.test_inference import test_inference
from edgeml.tests.test_actor import test_actor

if __name__ == '__main__':
    test_actor()
    test_inference()
    test_trainer()
    print(" [EdgeML] All tests passed!")
