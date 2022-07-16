"""uses ffmpeg to read frames from video."""
import cv2
import random
import ffmpeg
import numpy as np

from .shared_queue import SharedQueue


QUALITY = "360p"


def read_vids(vids, worker_id, take_every_nth, resize_size, queue_export):
    """
    Reads list of videos, saves frames to /dev/shm, and passes reading info through
    multiprocessing queue

    Input:
      vids - list of videos (either path or youtube link)
      queue - SharedQueue used to pass frames
      take_every_nth - offset between frames of video (to lower FPS)
      resize_size - new pixel height and width of resized frame
    """
    queue = SharedQueue.from_export(*queue_export)
    fps = int(25 / take_every_nth)

    def get_frames(vid):
        cap = cv2.VideoCapture(vid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        nw, nh = (-1, 224) if width > height else (224, -1)

        dst_name = vid[:-4].split("/")[-1] + ".npy"

        try:
            out, _ = (
                ffmpeg.input(vid)
                .filter("fps", fps=fps)
                .filter("scale", nw, nh)
                .filter("crop", w=224, h=224)
                .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="error")
                .run(capture_stdout=True)
            )
        except ffmpeg._run.Error:  # pylint: disable=protected-access
            print(f"Error: couldn't read video {vid}")
            return

        frame_count = int(len(out) / (224 * 224 * 3))  # can do this since dtype = np.uint8 (byte)
        vid_frames = np.frombuffer(out, np.uint8).reshape((frame_count, 224, 224, 3))
        queue.put(vid_frames, dst_name)

    random.Random(worker_id).shuffle(vids)
    for vid in vids:
        get_frames(vid)
