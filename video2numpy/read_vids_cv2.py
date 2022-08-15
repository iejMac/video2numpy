"""uses opencv to read frames from video."""
import cv2
import time
import numpy as np
import random

from .resizer import Resizer
from .shared_queue import SharedQueue
from .utils import handle_url


def read_vids(vid_refs, worker_id, take_every_nth, resize_size, batch_size, queue_export):
    """
    Reads list of videos, saves frames to Shared Queue

    Input:
      vid_refs - list of videos (either path or youtube link) and their references
      worker_id - unique ID of worker
      take_every_nth - offset between frames of video (to lower FPS)
      resize_size - new pixel height and width of resized frame
      batch_size - max length of frame sequence to put on shared_queue (-1 = no max).
      queue_export - SharedQueue export used re-create SharedQueue object in worker
    """
    queue = SharedQueue.from_export(*queue_export)
    t0 = time.perf_counter()
    print(f"Worker #{worker_id} starting processing {len(vid_refs)} videos")

    def get_frames(vid, ref):
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

        if len(video_frames) == 0:
            print(f"Warning: {vid} contained 0 frames")
            return

        np_frames = np.array(video_frames)
        f_ct = np_frames.shape[0]
        pad_by = 0
        if batch_size != -1:
            pad_by = (batch_size - f_ct % batch_size) % batch_size
            np_frames = np.pad(np_frames, ((0, pad_by), (0, 0), (0, 0), (0, 0)))
            np_frames = np_frames.reshape((-1, batch_size, resize_size, resize_size, 3))

        info = {
            "reference": ref,
            "dst_name": dst_name,
            "pad_by": pad_by,
        }
        queue.put(np_frames, info)

        if file is not None:  # for python files that need to be closed
            file.close()

    random.Random(worker_id).shuffle(vid_refs)
    for vid, ref in vid_refs:
        try:
            get_frames(vid, ref)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error: Video {vid} failed with message - {e}")
    tf = time.perf_counter()
    print(f"Worker #{worker_id} done processing {len(vid_refs)} videos in {tf-t0}[s]")
