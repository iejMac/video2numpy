"""uses opencv to read frames from video."""
import cv2
import random
import numpy as np
import ffmpeg
import librosa
import io

from .resizer import Resizer
from .shared_queue import SharedQueue
from .utils import handle_youtube

SAMPLING_RATE = 48000


def read_vids(vids, 
             worker_id, 
             take_every_nth, 
             resize_size, 
             batch_size, 
             queue_export_video,
             queue_export_audio,
             modalities = ["video", "audio"]
             ):
    """
    Reads list of videos, saves frames to Shared Queue,
    or optionally, save audios to Shared Queue. 

    Input:
      vids - list of videos (either path or youtube link)
      worker_id - unique ID of worker
      take_every_nth - offset between frames of video (to lower FPS)
      resize_size - new pixel height and width of resized frame
      batch_size - max length of frame sequence to put on shared_queue (-1 = no max).
      queue_export_video  - SharedQueue export for video frames, used re-create SharedQueue object in worker
      queue_export_audio  - SharedQueue export for audio, used re-create SharedQueue object in worker
      output_dir - directory to temporarily store audio files. All the audio files will be deleted 
            immidiately after they are converted to numpy array. 
      modalities - list of modalities to convert to numpy ndarray, ["video", "audio"] as default.
    """
    queue_video = SharedQueue.from_export(*queue_export_video)
    queue_audio = SharedQueue.from_export(*queue_export_audio)

    def get_frames(vid):
        if not vid.endswith(".mp4"):
            load_vid, dst_name = handle_youtube(vid, "video")
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


    def get_audio(vid):
        if not vid.endswith(".wav"):
            load_vid, dst_name = handle_youtube(vid, "audio")
        else:
            load_vid, dst_name = vid, vid[:-4].split("/")[-1] + ".wav"
        
        out, _ = (
            ffmpeg.input(load_vid)
            .output(f"pipe:", format="wav", ac = 1, ar = 48000)
            .run(capture_stdout=True)
        )

        np_audio, ar = librosa.core.load(io.BytesIO(out), sr = SAMPLING_RATE)
        

        info = {
            "dst_name": dst_name,
            "pad_by": 0,
        }

        queue_audio.put(np_audio, info)




    random.Random(worker_id).shuffle(vids)

    for vid in vids:
        if "video" in modalities:
            get_frames(vid)
        if "audio" in modalities:
            get_audio(vid)
        
