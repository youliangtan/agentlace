#!/usr/bin/env python3

import hashlib
import pickle
import sys
import numpy as np
import cv2

def mat_to_jpeg(img):
    """Compresses a numpy array into a JPEG byte array."""
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

def jpeg_to_mat(buf):
    """Decompresses a JPEG byte array into a numpy array."""
    return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)

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
