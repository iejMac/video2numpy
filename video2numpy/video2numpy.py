"""save frames from videos as numpy arrays"""
import os
import numpy as np

from .frame_reader import FrameReader


def video2numpy(src, dest="", take_every_nth=1, resize_size=224, workers=1, memory_size=4):
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
    workers:
        int: number of workers used to read videos
    memory_size:
        int: number of GB of shared memory used for reading, use larger shared memory for more videos
    """
    if isinstance(src, str):
        if src.endswith(".txt"):  # list of mp4s or youtube links
            with open(src, "r", encoding="utf-8") as f:
                fnames = [fn[:-1] for fn in f.readlines()]
        else:  # mp4 or youtube link
            fnames = [src]
    else:
        fnames = src

    batch_size = -1
    reader = FrameReader(fnames, take_every_nth, resize_size, batch_size, workers, memory_size)
    reader.start_reading()

    for vid_frames, info in reader:
        dst_name = info["dst_name"]
        save_pth = os.path.join(dest, dst_name)
        np.save(save_pth, vid_frames)
