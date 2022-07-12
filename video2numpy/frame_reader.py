"""reader - uses a reader function to read frames from videos"""
import numpy as np

from multiprocessing import shared_memory, SimpleQueue, Process

from .read_vids_cv2 import read_vids


class FrameReader:
    """
    Iterates over frame blocks returned by read_vids function
    """

    def __init__(
        self,
        fnames,
        chunk_size=1,
        take_every_nth=1,
        resize_size=224,
        auto_release=True,
    ):
        """
        Input:
          fnames - list with youtube links or paths to mp4 files
          chunk_size - how many videos to process at once
          take_every_nth - offset between frames we take
          resize_size - pixel height and width of target output shape
          auto_release - FrameReader iterator automatically releases shm buffers in next
                         iteration. This means the returned frame block or any slices
                         of it won't work in iterations following the one where it was returned.
                         If you plan on using it out of the iteration set this to False and
                         remember to keep a reference to the array and manually deallocate it
                         by calling release_memory once you're done.
        """
        self.auto_release = auto_release
        self.info_q = SimpleQueue()
        self.read_proc = Process(target=read_vids, args=(fnames, self.info_q, chunk_size, take_every_nth, resize_size))

        self.empty = False
        self.shms = []

    def __iter__(self):
        return self

    def __next__(self):
        if not self.empty:
            info = self.info_q.get()

            if isinstance(info, str):
                self.finish_reading()
                raise StopIteration

            if self.auto_release and len(self.shms) > 0:
                last_shm = self.shms.pop(0)
                last_shm.close()
                last_shm.unlink()

            shm = shared_memory.SharedMemory(name=info["shm_name"])
            block = np.ndarray(info["full_shape"], dtype=np.uint8, buffer=shm.buf)

            self.shms.append(shm)  # close and unlink when done using blocks

            return block, info["ind_dict"]
        raise StopIteration

    def start_reading(self):
        self.empty = False
        self.read_proc.start()

    def finish_reading(self):
        self.empty = True
        self.read_proc.join()
        if self.auto_release:
            self.release_memory()

    def release_memory(self):
        for shm in self.shms:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                print(f"Warning: Tried unlinking shared_memory block '{shm.name}' but file wasn't found")
        self.shms = []
