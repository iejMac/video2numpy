"""uses opencv to read frames from video."""
import cv2
import numpy as np
import random

from .resizer import Resizer
from .shared_queue import SharedQueue
from .utils import handle_url


def read_vids(vids, worker_id, take_every_nth, resize_size, batch_size, queue_export):
    """
    Reads list of videos, saves frames to Shared Queue

    Input:
      vids - list of videos (either path or youtube link)
      worker_id - unique ID of worker
      take_every_nth - offset between frames of video (to lower FPS)
      resize_size - new pixel height and width of resized frame
      batch_size - max length of frame sequence to put on shared_queue (-1 = no max).
      queue_export - SharedQueue export used re-create SharedQueue object in worker
    """
    queue = SharedQueue.from_export(*queue_export)

    def get_frames(vid):
        # TODO: better way of testing if vid is url
        if vid.startswith("http://") or vid.startswith("https://"):
            load_vid, file, dst_name = handle_url(vid)
        else:
            load_vid, file, dst_name = vid, None, vid[:-4].split("/")[-1] + ".npy"

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
            ret = cap.grab()
            if ret and (ind % take_every_nth == 0):
                ret, frame = cap.retrieve()
                frame = resizer(frame)
                video_frames.append(frame)
            ind += 1

        np_frames = np.array(video_frames)
        f_ct = np_frames.shape[0]
        pad_by = 0
        if batch_size != -1:
            pad_by = (batch_size - f_ct % batch_size) % batch_size
            np_frames = np.pad(np_frames, ((0, pad_by), (0, 0), (0, 0), (0, 0)))
            np_frames = np_frames.reshape((-1, batch_size, resize_size, resize_size, 3))

        info = {
            "dst_name": dst_name,
            "pad_by": pad_by,
        }
        queue.put(np_frames, info)

        if file is not None:  # for python files that need to be closed
            file.close()

    random.Random(worker_id).shuffle(vids)
    for vid in vids:
        try:
            get_frames(vid)
        except:  # pylint: disable=bare-except
            print(f"Error: Video {vid} failed")
