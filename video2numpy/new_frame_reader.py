"""reader - uses a reader function to read frames from videos"""
import numpy as np

from multiprocessing import shared_memory, SimpleQueue, Process

from .read_vids_sq_cv2 import read_vids


class FrameReader:
    """
    Iterates over frame blocks returned by read_vids function
    """

    def __init__(
        self,
        fnames,
        take_every_nth=1,
        resize_size=224,
        workers=8,
    ):
        """
        Input:
          fnames - list with youtube links or paths to mp4 files.
          chunk_size - how many videos to process at once.
          take_every_nth - offset between frames we take.
          resize_size - pixel height and width of target output shape.
          workers - number of Processes to distribute video reading to.
        """




    def __iter__(self):
        return self

    def __next__(self):
        pass

    def start_reading(self):
        pass

    def finish_reading(self):
        pass

    def release_memory(self):
        pass
