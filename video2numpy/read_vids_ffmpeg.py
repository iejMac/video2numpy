"""uses ffmpeg to read frames from video."""
import cv2
import ffmpeg
import numpy as np

from multiprocessing import shared_memory
from multiprocessing.pool import ThreadPool


QUALITY = "360p"
THREAD_COUNT = 12


def read_vids(vids, queue, chunk_size=1, take_every_nth=1, resize_size=224):
    """
    Reads list of videos, saves frames to /dev/shm, and passes reading info through
    multiprocessing queue

    Input:
      vids - list of videos (either path or youtube link)
      queue - multiprocessing queue used to pass frame block information
      chunk_size - size of chunk of videos to take for parallel reading
      take_every_nth - offset between frames of video (to lower FPS)
      resize_size - new pixel height and width of resized frame
    """

    while len(vids) > 0:
        vid_chunk = vids[:chunk_size]
        vids = vids[chunk_size:]

        frams = {}

        fps = int(25 / take_every_nth)

        with ThreadPool(THREAD_COUNT) as pool:

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
                except ffmpeg._run.Error:
                    print(f"Error: couldn't read video {vid}")
                    return

                frame_count = int(len(out) / (224 * 224 * 3))  # can do this since dtype = np.uint8 (byte)

                vid = np.frombuffer(out, np.uint8).reshape((frame_count, 224, 224, 3))
                frams[dst_name] = vid

            for _ in pool.imap_unordered(get_frames, vid_chunk):
                pass

        ind_dict = {}
        frame_count = 0
        for k, v in frams.items():
            ind_dict[k] = (frame_count, frame_count + len(v))
            frame_count += len(v)

        full_shape = (frame_count, resize_size, resize_size, 3)

        mem_size = frame_count * full_shape[0] * full_shape[1] * full_shape[2]
        shm = shared_memory.SharedMemory(create=True, size=mem_size)

        in_arr = np.ndarray(full_shape, dtype=np.uint8, buffer=shm.buf)
        for k, v in frams.items():
            i0, it = ind_dict[k]
            if it > i0:
                in_arr[i0:it] = v

        info = {
            "ind_dict": ind_dict,
            "shm_name": shm.name,
            "full_shape": full_shape,
        }
        shm.close()
        queue.put(info)

    queue.put("DONE_READING")
