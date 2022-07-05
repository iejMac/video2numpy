import os
import glob
import time
import numpy as np

from multiprocessing import SimpleQueue, Process, shared_memory
from torch.utils.data import DataLoader

from reader import FrameReader

if __name__ == "__main__":
  print("Starting program...")

  fnames = glob.glob("k700/*.mp4")

  print(f"Reading {len(fnames)} videos...")

  VID_CHUNK_SIZE = 4
  take_every_nth = 1

  reader = FrameReader(fnames, VID_CHUNK_SIZE, take_every_nth)
  print(f"Starting reading process...")

  t0 = time.perf_counter()
  reader.start_reading()

  all_frames = 0

  for block, ind_dict in reader:
    all_frames += block.shape[0]

  reader.finish_reading()
  
  read_time = time.perf_counter() - t0
  print(f"FPS: {all_frames/read_time}")
