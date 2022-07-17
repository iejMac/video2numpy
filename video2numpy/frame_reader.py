"""reader - uses a reader function to read frames from videos"""
import multiprocessing

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
        workers=1,
        memory_size=4,
    ):
        """
        Input:
          vids - list with youtube links or paths to mp4 files.
          chunk_size - how many videos to process at once.
          take_every_nth - offset between frames we take.
          resize_size - pixel height and width of target output shape.
          workers - number of Processes to distribute video reading to.
          memory_size - number of GB of shared_memory
        """

        memory_size_b = memory_size * 1024**3
        shared_frames = memory_size_b // (256**2 * 3)
        self.shared_queue = SharedQueue.from_shape([shared_frames, resize_size, resize_size, 3])

        div_vids = [vids[int(len(vids) * i / workers) : int(len(vids) * (i + 1) / workers)] for i in range(workers)]
        self.procs = [
            multiprocessing.Process(
                args=(work, worker_id, take_every_nth, resize_size, self.shared_queue.export()),
                daemon=True,
                target=read_vids,
            )
            for worker_id, work in enumerate(div_vids)
        ]

    def __iter__(self):
        return self

    def __next__(self):
        if self.shared_queue or any(p.is_alive() for p in self.procs):
            frames, name = self.shared_queue.get()
            return frames, name
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
