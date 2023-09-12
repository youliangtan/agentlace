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
