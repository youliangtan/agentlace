#!/usr/bin/env python3

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Any
from collections import deque
from threading import Lock

from abc import abstractmethod

##############################################################################


class DataStoreBase:
    @abstractmethod
    def latest_data_id() -> Any:
        """Return the id of the latest data"""
        pass

    @abstractmethod
    def get_latest_data(self, from_id: Any) -> Tuple[list, dict]:
        """
        provide the all data from the given id
            :return indices of the updated data within the store
                    and it's corresponding data
        """
        pass

    @abstractmethod
    def update_data(self, indices: list, data: dict):
        """with the indices and data, update the latest data"""
        pass

    @abstractmethod
    def __len__(self):
        """Length of the valid data in the store"""
        pass

##############################################################################


class QueuedDataStore(DataStoreBase):
    """A simple queue-based data store."""

    def __init__(self, capacity: int):
        self.queue = deque(maxlen=capacity)
        self.latest_seq_id = -1
        self._lock = Lock()  # Mutex

    def latest_data_id(self) -> int:
        return self.latest_seq_id

    def insert(self, data: Any):
        self._lock.acquire()
        self.latest_seq_id += 1
        self.queue.append((self.latest_seq_id, data))
        self._lock.release()

    def get_latest_data(self, from_id: int) -> Tuple[List[int], Dict]:
        indices = []
        output_data = {"seq_id": [], "data": []}

        self._lock.acquire()
        for idx, (seq_id, data) in enumerate(self.queue):
            if seq_id > from_id:
                indices.append(idx)
                output_data["seq_id"].append(seq_id)
                output_data["data"].append(data)
        self._lock.release()
        return indices, output_data

    def update_data(self, indices: List[int], data: Dict):
        self._lock.acquire()
        assert len(indices) == len(data["seq_id"]) == len(data["data"])
        for i, idx in enumerate(indices):
            _id, _data = data["seq_id"][i], data["data"][i]
            if idx < len(self.queue):
                self.queue[idx] = (_id, _data)
            else:
                self.queue.append((_id, _data))
        # the last data is always the latest
        if len(self.queue) > 0:
            self.latest_seq_id = self.queue[-1][0]
        self._lock.release()

    def __len__(self):
        self._lock.acquire()
        length = len(self.queue)
        self._lock.release()
        return length
