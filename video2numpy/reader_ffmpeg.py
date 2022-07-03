"""same as reader.py but uses ffmpeg"""
import cv2
import ffmpeg
import numpy as np

from multiprocessing import shared_memory
from multiprocessing.pool import ThreadPool


QUALITY = "360p"
MAX_THREAD_COUNT = 10
POSTPROC_SHAPE = (224, 224, 3)


def read_vids(vids, queue, comp_queue, chunk_size=1, take_every_nth=1):
    """same as reader.py"""
    shms = []

    while len(vids) > 0:
        vid_chunk = vids[:chunk_size]
        vids = vids[chunk_size:]

        frams = {}

        fps = int(25 / take_every_nth)

        with ThreadPool(MAX_THREAD_COUNT) as pool:

            def get_frames(video):

                cap = cv2.VideoCapture(video)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                nw, nh = (-1, 224) if width > height else (224, -1)

                dst_name = video[:-4].split("/")[-1] + ".npy"

                out, _ = (
                    ffmpeg.input(video)
                    .filter("fps", fps=fps)
                    .filter("scale", nw, nh)
                    .filter("crop", w=224, h=224)
                    .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="error")
                    .run(capture_stdout=True)
                )
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

        full_shape = (frame_count, 224, 224, 3)

        mem_size = frame_count * full_shape[0] * full_shape[1] * full_shape[2]
        shm = shared_memory.SharedMemory(create=True, size=mem_size)

        in_arr = np.ndarray(full_shape, dtype=np.uint8, buffer=shm.buf)
        for k, v in frams.items():
            i0, it = ind_dict[k]
            in_arr[i0:it] = v

        info = {
            "ind_dict": ind_dict,
            "shm_name": shm.name,
            "frame_count": frame_count,
        }

        print(f"Put {in_arr.shape} on queue")
        shms.append(shm)
        shm.close()

        queue.put(info)

    queue.put("DONE_READING")

    # Wait for DONE_MAPPING
    msg = comp_queue.get()
    if msg != "DONE_MAPPING":
        print("Error: Message wrong message received")

    for shm in shms:
        shm.unlink()
