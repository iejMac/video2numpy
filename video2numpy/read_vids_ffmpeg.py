"""uses ffmpeg to read frames from video."""
import cv2
import random
import ffmpeg
import numpy as np

from .shared_queue import SharedQueue
from .utils import handle_youtube


QUALITY = "360p"


def read_vids(vids, worker_id, take_every_nth, resize_size, queue_export):
    """
    Reads list of videos, saves frames to SharedQueue

    Input:
      vids - list of videos (either path or youtube link)
      worker_id - unique ID of worker
      take_every_nth - offset between frames of video (to lower FPS)
      resize_size - new pixel height and width of resized frame
      queue_export - SharedQueue export used re-create SharedQueue object in worker
    """
    queue = SharedQueue.from_export(*queue_export)
    fps = int(25 / take_every_nth)

    def get_frames(vid):
        if not vid.endswith(".mp4"):
            load_vid, dst_name = handle_youtube(vid)
        else:
            load_vid, dst_name = vid, vid[:-4].split("/")[-1] + ".npy"

        cap = cv2.VideoCapture(load_vid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        nw, nh = (-1, resize_size) if width > height else (resize_size, -1)

        try:
            out, _ = (
                ffmpeg.input(load_vid)
                .filter("fps", fps=fps)
                .filter("scale", nw, nh)
                .filter("crop", w=resize_size, h=resize_size)
                .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="error")
                .run(capture_stdout=True)
            )
        except ffmpeg._run.Error:  # pylint: disable=protected-access
            print(f"Error: couldn't read video {vid}")
            return

        frame_count = int(len(out) / (resize_size * resize_size * 3))  # can do this since dtype = np.uint8 (byte)
        vid_frames = np.frombuffer(out, np.uint8).reshape((frame_count, resize_size, resize_size, 3))
        queue.put(vid_frames, dst_name)

    random.Random(worker_id).shuffle(vids)
    for vid in vids:
        get_frames(vid)
