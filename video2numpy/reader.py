"""reader - uses cv2 to read frames from videos"""
import cv2
import youtube_dl
import numpy as np

from multiprocessing import shared_memory, SimpleQueue, Process
from multiprocessing.pool import ThreadPool

from .resize import Resizer


QUALITY = "360p"
THREAD_COUNT = 12


class FrameReader:
    """
    Iterates over frame blocks returned by read_vids function
    """

    def __init__(
        self,
        fnames,
        chunk_size=1,
        take_every_nth=1,
        resize_size=224,
        auto_release=True,
    ):
        """
        Describe args
        """
        self.auto_release = auto_release
        self.info_q = SimpleQueue()
        self.read_proc = Process(target=read_vids, args=(fnames, self.info_q, chunk_size, take_every_nth, resize_size))

        self.empty = False
        self.shms = []

    def __iter__(self):
        return self

    def __next__(self):
        if not self.empty:
            info = self.info_q.get()

            if isinstance(info, str):
                self.finish_reading()
                raise StopIteration

            shm = shared_memory.SharedMemory(name=info["shm_name"])
            block = np.ndarray(info["full_shape"], dtype=np.uint8, buffer=shm.buf)

            self.shms.append(shm)  # close and unlink when done using blocks

            return block, info["ind_dict"]
        raise StopIteration

    def start_reading(self):
        self.empty = False
        self.read_proc.start()

    def finish_reading(self):
        self.empty = True
        self.read_proc.join()
        if self.auto_release:
            self.release_memory()

    def release_memory(self):
        for shm in self.shms:
            shm.close()
            shm.unlink()
        self.shms = []


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

        with ThreadPool(THREAD_COUNT) as pool:

            def generate_frames(vid):

                video_frames = []

                if not vid.endswith(".mp4"):  # youtube link
                    ydl_opts = {}
                    ydl = youtube_dl.YoutubeDL(ydl_opts)
                    info = ydl.extract_info(vid, download=False)
                    formats = info.get("formats", None)
                    f = None
                    for f in formats:
                        if f.get("format_note", None) != QUALITY:
                            continue
                        break

                    cv2_vid = f.get("url", None)

                    dst_name = info.get("id") + ".npy"
                else:
                    cv2_vid = vid
                    dst_name = vid[:-4].split("/")[-1] + ".npy"

                cap = cv2.VideoCapture(cv2_vid)  # pylint: disable=I1101

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

                frams[dst_name] = video_frames

            for _ in pool.imap_unordered(generate_frames, vid_chunk):
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
