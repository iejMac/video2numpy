"""uses opencv to read frames from video."""
import cv2
import random
import numpy as np

from .resizer import Resizer
from .shared_queue import SharedQueue
from .utils import handle_youtube
from .utils import extract_audio_from_url


def read_vids(vids, 
             worker_id, 
             take_every_nth, 
             resize_size, 
             batch_size, 
             queue_export_video,
             queue_export_audio,
             output_dir,
             no_audio,
             no_video):
    """
    Reads list of videos, saves frames to Shared Queue,
    or optionally, save audios to Shared Queue. 

    Input:
      vids - list of videos (either path or youtube link)
      worker_id - unique ID of worker
      take_every_nth - offset between frames of video (to lower FPS)
      resize_size - new pixel height and width of resized frame
      batch_size - max length of frame sequence to put on shared_queue (-1 = no max).
      queue_export_video  - SharedQueue export used re-create SharedQueue object in worker
      queue_export_audio  - SharedQueue export used re-create SharedQueue object in worker
      output_dir - directory to temporarily store audio files. All the audio files will be deleted 
            immidiately after they are converted to numpy array. 
      no_audio - boolean, if True, do not extract audio
      no_video - boolean, if True, do not extract video
    """
    queue_video = SharedQueue.from_export(*queue_export_video)
    queue_audio = SharedQueue.from_export(*queue_export_audio)

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
        queue_video.put(np_frames, info)

    random.Random(worker_id).shuffle(vids)

    for vid in vids:
        if not no_video:
            get_frames(vid)
        if not no_audio:
            np_array, dst_name = extract_audio_from_url(vid,output_dir) 
            info = {
                "dst_name": dst_name,
                "pad_by": 0,
            } 
            queue_audio.put(np_array, info)
