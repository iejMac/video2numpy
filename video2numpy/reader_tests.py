"""testing various video reading methods"""
import time
import glob

import numpy as np

import ffmpeg
import cv2

from multiprocessing.pool import ThreadPool
from multiprocessing import Process


MAX_THREAD_COUNT = 10

VIDS = glob.glob("/home/iejmac/test_vids/*.mp4")
VIDS = VIDS[:200]
print(len(VIDS))

# default samplign is bicubic (https://trac.ffmpeg.org/wiki/Scaling)
def test_ffmpeg(fps):
    """tests reading using python-ffmpeg"""
    frams = {}

    with ThreadPool(MAX_THREAD_COUNT) as pool:

        def get_frames(video):

            cap = cv2.VideoCapture(video)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            nw, nh = (-1, 224) if width > height else (224, -1)

            dst_name = video[:-4].split("/")[-1] + ".npy"

            out, _ = (
                ffmpeg.input(video, vcodec="h264")
                .filter("fps", fps=fps)
                .filter("scale", nw, nh)
                .filter("crop", w=224, h=224)
                .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="error")
                .run(capture_stdout=True)
            )
            frame_count = int(len(out) / (224 * 224 * 3))  # can do this since dtype = np.uint8 (byte)

            vid = np.frombuffer(out, np.uint8).reshape((frame_count, 224, 224, 3))

            frams[dst_name] = vid

        for _ in pool.imap_unordered(get_frames, VIDS):
            pass

    frame_count = 0
    for v in frams.values():
        frame_count += len(v)
    print(frame_count)


def test_cv2(take_every_nth):
    """tests reading using cv2"""
    frams = {}

    with ThreadPool(MAX_THREAD_COUNT) as pool:

        def get_frames(video):
            cap = cv2.VideoCapture(video)
            frames = []
            dst_name = video[:-4].split("/")[-1] + ".npy"

            ret = True
            ind = 0

            shape = None
            while ret:
                ret, frame = cap.read()

                if ret and (ind % take_every_nth == 0):

                    if shape is None:
                        cur_shape = list(frame.shape)[:2]
                        sm_ind, bg_ind = (0, 1) if cur_shape[0] < cur_shape[1] else (1, 0)
                        ratio = cur_shape[sm_ind] / 224
                        n_shape = cur_shape
                        n_shape[sm_ind] = 224
                        n_shape[bg_ind] = int(n_shape[bg_ind] / ratio)
                        shape = tuple(n_shape)

                    # Resize:
                    frame = cv2.resize(frame, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
                    # Center crop:
                    my = int((shape[0] - 224) / 2)
                    mx = int((shape[1] - 224) / 2)

                    frame = frame[my : frame.shape[0] - my, mx : frame.shape[1] - mx]
                    frame = frame[:224, :224]

                    frames.append(frame)

                ind += 1

            frams[dst_name] = frames

        for _ in pool.imap_unordered(get_frames, VIDS):
            pass

    frame_count = 0
    for v in frams.values():
        frame_count += len(v)
    print(frame_count)


def test_method(method, args):
    """generic func for displaying info about method performance"""
    start_time = time.perf_counter()

    meth_proc = Process(target=method, args=args)

    meth_proc.start()
    meth_proc.join()

    end_time = time.perf_counter() - start_time

    print(f"REPORT FOR {method.__name__}")
    print(f"Read time: {end_time}")
    # print(f"Read FPS: {frame_count / end_time}")
    print("===============================")


if __name__ == "__main__":
    FPS = 1
    ten = int(25 / FPS)

    test_method(test_ffmpeg, (FPS,))
    test_method(test_cv2, (ten,))
