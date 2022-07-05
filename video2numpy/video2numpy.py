import os
import numpy as np

from .reader import FrameReader


VID_CHUNK_SIZE = 2
QUALITY = "360p"


def video2numpy(src, dest='', take_every_nth=1):
    """
        TODO: write docstring
    """
    if isinstance(src, str):
        if src.endswith(".txt"):  # list of mp4s or youtube links
            with open(src, "r", encoding="utf-8") as f:
                fnames = [fn[:-1] for fn in f.readlines()]
        else:  # mp4 or youtube link
            fnames = [src]
    else:
        fnames = src

    reader = FrameReader(fnames, VID_CHUNK_SIZE, take_every_nth)
    reader.start_reading()

    for block, ind_dict in reader:
      for dst_name, inds in ind_dict.items():
        i0, it = inds
        frames = block[i0:it]
        save_pth = os.path.join(dest, dst_name)
        np.save(save_pth, frames)
