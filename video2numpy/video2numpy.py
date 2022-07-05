"""save frames from videos as numpy arrays"""
import os
import numpy as np

from .reader import FrameReader


VID_CHUNK_SIZE = 2
QUALITY = "360p"


def video2numpy(src, dest="", take_every_nth=1, resize_size=224):
    """
    Read frames from videos and save as numpy arrays

    Input:
      src:
        str: path to mp4 file
        str: youtube link
        str: path to txt file with multiple mp4's or youtube links
        list: list with multiple mp4's or youtube links
      dest:
        str: directory where to save frames to
        None: dest = src + .npy
      take_every_nth:
        int: only take every nth frame
      resize_size:
        int: new pixel height and width of resized frame
    """
    if isinstance(src, str):
        if src.endswith(".txt"):  # list of mp4s or youtube links
            with open(src, "r", encoding="utf-8") as f:
                fnames = [fn[:-1] for fn in f.readlines()]
        else:  # mp4 or youtube link
            fnames = [src]
    else:
        fnames = src

    reader = FrameReader(fnames, VID_CHUNK_SIZE, take_every_nth, resize_size)
    reader.start_reading()

    for block, ind_dict in reader:
        for dst_name, inds in ind_dict.items():
            i0, it = inds
            frames = block[i0:it]
            save_pth = os.path.join(dest, dst_name)
            np.save(save_pth, frames)
