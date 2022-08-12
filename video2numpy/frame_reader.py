"""reader - uses a reader function to read frames from videos"""
import multiprocessing
import random

from .read_vids_cv2 import read_vids
from .shared_queue import SharedQueue


class FrameReader:
    """
    Iterates over frame blocks returned by read_vids function
    """

    def __init__(
        self,
        vids,
        take_every_nth=1,
        resize_size=224,
        batch_size=-1,
        workers=1,
        memory_size=4,
    ):
        """
        Input:
          vids - list with youtube links or paths to mp4 files.
          chunk_size - how many videos to process at once.
          take_every_nth - offset between frames we take.
          resize_size - pixel height and width of target output shape.
          batch_size - max length of frame sequence to put on shared_queue (-1 = no max).
          workers - number of Processes to distribute video reading to.
          memory_size - number of GB of shared_memory
        """
        self.n_vids = len(vids)
        random.shuffle(vids)  # shuffle videos so each shard has approximately equal sum of video lengths

        memory_size_b = int(memory_size * 1024**3)  # GB -> bytes
        shared_blocks = memory_size_b // (resize_size**2 * 3 * (1 if batch_size == -1 else batch_size))
        dim12 = (shared_blocks,) if batch_size == -1 else (shared_blocks, batch_size)
        self.shared_queue = SharedQueue.from_shape([*dim12, resize_size, resize_size, 3])

        div_vids = [vids[int(len(vids) * i / workers) : int(len(vids) * (i + 1) / workers)] for i in range(workers)]

        self.procs = [
            multiprocessing.Process(
                args=(work, worker_id, take_every_nth, resize_size, batch_size, self.shared_queue.export()),
                daemon=True,
                target=read_vids,
            )
            for worker_id, work in enumerate(div_vids)
        ]

    def __len__(self):
        return self.n_vids

    def __iter__(self):
        return self

    def __next__(self):
        if self.shared_queue or any(p.is_alive() for p in self.procs):
            frames, info = self.shared_queue.get()
            return frames, info
        self.finish_reading()
        self.release_memory()
        raise StopIteration

    def start_reading(self):
        for p in self.procs:
            p.start()

    def finish_reading(self):
        for p in self.procs:
            p.join()

    def release_memory(self):
        self.shared_queue.frame_mem.unlink()
        self.shared_queue.frame_mem.close()
