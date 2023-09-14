#!/usr/bin/env python3

import hashlib
import pickle
import sys

def compute_hash(obj):
    pickle_str = pickle.dumps(obj)
    # Compute MD5 hash of the string
    return hashlib.md5(pickle_str).hexdigest()

def print_size(obj):
    size = sys.getsizeof(obj)
    mb_size = size / 1024 ** 2
    print(f"The size of the object is {mb_size} MB")

def print_error(text):
    # Red color
    print(f"\033[91m{text}\033[00m")

def print_warning(text):
    # Yellow color
    print(f"\033[93m{text}\033[00m")
