import os
import glob
import numpy as np

from multiprocessing import SimpleQueue, Process, shared_memory
from torch.utils.data import DataLoader

from reader_new import read_vids, Reader

if __name__ == "__main__":
  print("Starting program...")

  fnames = glob.glob("k700/*.mp4")

  print(f"Reading {len(fnames)} videos...")

  VID_CHUNK_SIZE = 4
  take_every_nth = 10

  # vr_proc = Process(target=read_vids, args=(fnames, info_q, VID_CHUNK_SIZE, take_every_nth))
  reader = Reader(read_vids, fnames, VID_CHUNK_SIZE, take_every_nth)
  print(f"Starting reading process...")
  reader.start()

  while not reader.empty:
    block, ind_dict = reader.next_block()

    if block is not None:
      print(block.shape)

  reader.join()
