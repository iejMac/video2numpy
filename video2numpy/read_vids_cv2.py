"""uses ffmpeg to read frames from video."""
import cv2
import random
import ffmpeg
import numpy as np

from multiprocessing import shared_memory
from multiprocessing.pool import ThreadPool

from .resizer import Resizer
from .shared_queue import SharedQueue
from .utils import handle_youtube



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

    def get_frames(vid):
        if not vid.endswith(".mp4"):
            load_vid, dst_name = handle_youtube(vid)
        else:
            load_vid, dst_name = vid, vid[:-4].split("/")[-1] + ".npy"

        video_frames = []
        cap = cv2.VideoCapture(load_vid)  # pylint: disable=I1101

        if not cap.isOpened():
            print(f"Error: {vid} not opened")
            return

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_shape = [height, width, 3]

        resizer = Resizer(frame_shape, resize_size)

        ret = True
        ind = 0
        while ret:
            ret, frame = cap.read()
            if ret and (ind % take_every_nth == 0):
                frame = resizer(frame)
                video_frames.append(frame)
            ind += 1
        queue.put(np.array(video_frames), dst_name)

    random.Random(worker_id).shuffle(vids)
    for vid in vids:
        get_frames(vid)
