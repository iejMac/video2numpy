"""SharedQueue for saving arrays of video frames."""
import datetime
import time
import multiprocessing
import numpy as np
import threading
import typing

from multiprocessing.shared_memory import SharedMemory


# taken from: https://github.com/ClashLuke
class SharedQueue:
    """SharedQueue class."""

    frame_mem: SharedMemory
    frame: np.ndarray
    indices: list
    write_index_lock: threading.Lock
    read_index_lock: threading.Lock

    @classmethod
    def from_shape(cls, shape: typing.List[int]):
        """create SharedQueue from shape."""
        self = cls()
        frames = np.zeros(shape, dtype=np.uint8)
        self.frame_mem = SharedMemory(create=True, size=frames.nbytes)
        self.frame = np.ndarray(shape, dtype=np.uint8, buffer=self.frame_mem.buf)
        manager = multiprocessing.Manager()
        self.indices = manager.list()
        self.frame[:] = 0  # TODO: do you need this?
        self.write_index_lock = manager.Lock()
        self.read_index_lock = manager.Lock()
        return self

    @classmethod
    def from_export(cls, frame_name, frame_shape, indices, write_index_lock, read_index_lock):
        self = cls()
        self.frame_mem = SharedMemory(create=False, name=frame_name)
        self.frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=self.frame_mem.buf)
        self.indices = indices
        self.write_index_lock = write_index_lock
        self.read_index_lock = read_index_lock
        return self

    def export(self):
        return self.frame_mem.name, self.frame.shape, self.indices, self.write_index_lock, self.read_index_lock

    def get(self):
        while True:
            with self.read_index_lock:
                while not self:
                    time.sleep(1)
                info, start, end = self.indices.pop(0)
            return self.frame[start:end].copy(), info  # local clone, so share can be safely edited

    def _free_memory(self, size: int) -> typing.Optional[typing.Tuple[int, int, int]]:
        if not self:
            return 0, 0, size
        local_indices = list(self.indices)
        itr = zip([[{}, None, 0]] + local_indices, local_indices + [[{}, self.frame.shape[0], None]])  # type: ignore
        for i, ((_, _, prev_end), (_, start, _)) in enumerate(itr):
            if start - prev_end > size:
                return i, prev_end, prev_end + size  # type: ignore
        return None

    def _put_item(self, obj: np.ndarray, info: dict):
        batches = obj.shape[0]
        with self.write_index_lock:
            indices = self._free_memory(batches)
            if indices is None:
                return
            idx, start, end = indices
            self.indices.insert(idx, (info, start, end))
        self.frame[start:end] = obj[:]  # we simply assume that the synchronisation overheads make the reader slower

    def put(self, obj: np.ndarray, info: dict):
        """Put array on queue."""
        batches = obj.shape[0]
        max_size = self.frame.shape[0] // 4  # unrealistic that it'll fit if it takes up 25% of the memory
        if batches > max_size:
            for idx in range(0, batches, max_size):  # ... so we slice it up and feed in many smaller videos
                self.put(obj[idx : idx + max_size], info)
            return

        def _fits():
            return bool(self._free_memory(batches))

        # until new frames fit into memory
        waiting = 12
        while not _fits():
            time.sleep(5)
            waiting -= 1
        if not waiting:
            print(
                "Warning: waited for one minute for space to free up, but none found. Increase memory size to avoid "
                "fragmentation or implement defragmentation. Timestamp:",
                datetime.datetime.now(),
                flush=True,
            )
            return

        self._put_item(obj, info)

    def __bool__(self):
        return bool(self.indices)
