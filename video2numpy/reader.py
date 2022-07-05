"""reader - uses cv2 to read frames from videos"""
import cv2
import youtube_dl
import numpy as np

from multiprocessing import shared_memory, SimpleQueue, Process
from multiprocessing.pool import ThreadPool


QUALITY = "360p"
THREAD_COUNT = 12
POSTPROC_SHAPE = (224, 224, 3)
IMG_SIDE = 224


class Reader:
    """
    Iterates over frame blocks returned by read_vids function
    """
    def __init__(
        self,
        read_func,
        fnames,
        chunk_size=1,
        take_every_nth=1
    ):
        self.info_q = SimpleQueue()
        self.read_proc = Process(target=read_func, args=(fnames, self.info_q, chunk_size, take_every_nth))

        self.empty = False

    def __iter__(self):
        return self
    def __next__(self):
        if not self.empty:
            info = self.info_q.get()

            if isinstance(info, str):
                self.empty = True
                raise StopIteration

            shm = shared_memory.SharedMemory(name=info["shm_name"])
            block = np.ndarray(info["full_shape"], dtype=np.uint8, buffer=shm.buf)

            # Clean up shm
            shm.close()
            shm.unlink()

            return block, info["ind_dict"]
        raise StopIteration 

    def start(self):
        self.read_proc.start()
    def join(self):
        self.read_proc.join()


def read_vids(vids, queue, chunk_size=1, take_every_nth=1):
    """
    Reads list of videos, saves frames to /dev/shm, and passes reading info through
    multiprocessing queue

    Input:
      vids - list of videos (either path or youtube link)
      queue - multiprocessing queue used to pass frame block information
      chunk_size - size of chunk of videos to take for parallel reading
      take_every_nth - offset between frames of video (to lower FPS)
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

                ret = True
                frame_shape = None
                ind = 0
                while ret:
                    ret, frame = cap.read()

                    if ret and (ind % take_every_nth == 0):
                        # NOTE: HACKY
                        if frame_shape is None:
                            cur_shape = list(frame.shape)[:2]
                            sm_ind, bg_ind = (0, 1) if cur_shape[0] < cur_shape[1] else (1, 0)
                            ratio = cur_shape[sm_ind] / IMG_SIDE
                            n_shape = cur_shape
                            n_shape[sm_ind] = IMG_SIDE
                            n_shape[bg_ind] = max(int(n_shape[bg_ind] / ratio), IMG_SIDE)  # safety for rounding errors
                            frame_shape = tuple(n_shape)

                        # Resize:
                        frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_CUBIC)
                        # Center crop:
                        my = int((frame_shape[0] - IMG_SIDE) / 2)
                        mx = int((frame_shape[1] - IMG_SIDE) / 2)

                        frame = frame[my : frame.shape[0] - my, mx : frame.shape[1] - mx]
                        frame = frame[:IMG_SIDE, :IMG_SIDE]

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

        full_shape = (frame_count, *POSTPROC_SHAPE)

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
