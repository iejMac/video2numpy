import argparse
import copy
import datetime
import functools
import multiprocessing
import os
import pickle
import random
import shutil
import sys
import threading
import time
import traceback
import typing
import uuid
from multiprocessing.shared_memory import SharedMemory

import boto3
import ffmpeg
import gdown
import numpy as np
import requests
import tensorflow as tf
import torch
import youtube_dl
from omegaconf import OmegaConf

sys.path.append("./taming-transformers")
from taming.models.vqgan import GumbelVQ


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-worker", type=int, default=multiprocessing.cpu_count(),
                        help=f"Number of workers. Default is the number of CPU cores (={multiprocessing.cpu_count()})")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--model-base-path", type=str, default='/fsx/lucas',
                        help="Where model and config should be dowloaded to")
    parser.add_argument("--bucket", type=str, help="Name of the S3 bucket")
    parser.add_argument("--prefix", type=str, help="Prefix in the bucket")
    parser.add_argument("--batch", type=int, default=128, help="Number of images processed per 'computation step'")
    parser.add_argument("--tmp-dir", type=str, help="Local directory for temporary storage")
    parser.add_argument("--urls", type=str, help="Directory filled with JSON files full of URLs")
    parser.add_argument("--fps", type=int, default=1,
                        help="Number of (encoded) video frames per second of raw data (default=4)")
    parser.add_argument("--shared-memory", type=int, default=4, help="number of GB of shared memory")
    parser.add_argument("--tokens-per-file", type=int, default=2 ** 28, help="how big each file should roughly be")
    parser.add_argument("--video-downloaders", type=int, default=4,
                        help="Number of parallel video _information_ downloaders. Videos are always downloaded in "
                             "parallel, but downloading information about too many videos in parallel can lead to "
                             "errors and slow things down.")
    args = parser.parse_args()
    return args.cpu_worker, args.bucket, args.prefix, args.tmp_dir, args.urls, args.fps, \
           args.batch, args.gpus, args.model_base_path, args.shared_memory, args.tokens_per_file, args.video_downloaders


def try_except(fn: typing.Callable, default=None):
    def _fn(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            print(r"IGNORED EXCEPTION \/\/\/")
            print(fn, exc)
            traceback.print_exc()
            print("IGNORED EXCEPTION /\\/\\/\\")

        return default

    return _fn


@try_except
def get_video_urls(youtube_getter, youtube_base: str, url: str, lock: threading.Semaphore, target_image_size: int) -> \
        typing.List[dict]:
    # We have to lock this part because it can lead to errors if multiple thread try to scrape video Information at
    # the same time.
    with lock:
        info = youtube_getter.extract_info(youtube_base + url, download=False)
    if info is None or 'formats' not in info:
        return []
    video_urls = []
    for f in info['formats']:
        width = f.get('width')
        height = f.get('height')
        url = f.get('url')
        ext = f.get('ext')
        format_note = f.get('format_note')

        if any(x is None for x in (width, height, url, ext, format_note)):
            continue
        if any(not x for x in (width, height, url, ext)):
            continue
        if format_note == "tiny" or width <= target_image_size or height <= target_image_size:
            continue
        video_urls.append({'width': width, 'height': height, 'ext': f['ext'], 'url': f['url']})
    return sorted(video_urls, key=lambda x: (x['ext'] != 'mp4', x['width'], x['height']))


def get_video_frames(video_urls: typing.List[dict], target_image_size: int, target_fps: int) -> np.ndarray:
    filename = uuid.uuid4()
    path = str(filename)
    for video_url_idx, video_url in enumerate(video_urls):
        if os.path.exists(path):
            os.remove(path)

        url = video_url["url"]
        path = f"{filename}.{video_url['ext']}"

        try:
            with requests.get(url, stream=True) as r, open(path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        except Exception:  # Broken URL, next might work
            continue

        width = round(video_url["width"] * video_url["height"] / target_image_size)
        try:
            out, _ = ffmpeg.input(path, preset="ultrafast", threads=target_image_size // 40) \
                .filter("scale", w=width, h=target_image_size) \
                .filter("crop", w=target_image_size, h=target_image_size).filter("fps", target_fps) \
                .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="error").run(capture_stdout=True)
        except ffmpeg.Error:  # Broken Video, next might work
            continue

        if os.path.exists(path):
            os.remove(path)
        return np.frombuffer(out, np.uint8).reshape((-1, target_image_size, target_image_size, 3))


class SharedQueue:
    frame_mem: SharedMemory
    frame: np.ndarray
    indices: list
    write_index_lock: threading.Lock
    read_index_lock: threading.Lock

    @classmethod
    def from_shape(cls, shape: typing.List[int]):
        self = cls()
        frames = np.zeros(shape, dtype=np.uint8)
        self.frame_mem = SharedMemory(create=True, size=frames.nbytes)
        self.frame = np.ndarray(shape, dtype=np.uint8, buffer=self.frame_mem.buf)
        manager = multiprocessing.Manager()
        self.indices = manager.list()
        self.frame[:] = 0
        self.write_index_lock = manager.Lock()
        self.read_index_lock = manager.Lock()
        return self

    @classmethod
    def from_export(cls, frame_name, frame_shape, indices, write_index_lock, read_index_lock):
        self = cls()
        self.frame_mem = SharedMemory(create=False, name=frame_name)
        self.frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=self.frame_mem.buf)
        self.indices = indices
        self.write_index_lock = write_index_lock
        self.read_index_lock = read_index_lock
        return self

    def export(self):
        return self.frame_mem.name, self.frame.shape, self.indices, self.write_index_lock, self.read_index_lock

    def get(self):
        while True:
            with self.read_index_lock:
                while not self:
                    time.sleep(1)
                start, end = self.indices.pop(0)
            return self.frame[start:end].copy()  # local clone, so share can be safely edited

    def _free_memory(self, size: int) -> typing.Optional[typing.Tuple[int, int, int]]:
        if not self:
            return 0, 0, size
        local_indices = list(self.indices)
        itr = zip([[None, 0]] + local_indices, local_indices + [[self.frame.shape[0], None]])
        for i, ((_, prev_end), (start, _)) in enumerate(itr):
            if start - prev_end > size:
                return i, prev_end, prev_end + size

    def _put_item(self, obj: np.ndarray):
        batches = obj.shape[0]
        with self.write_index_lock:
            indices = self._free_memory(batches)
            if indices is None:
                return
            idx, start, end = indices
            self.indices.insert(idx, (start, end))
        self.frame[start:end] = obj[:]  # we simply assume that the synchronisation overheads make the reader slower

    def put(self, obj: np.ndarray):
        batches = obj.shape[0]
        max_size = self.frame.shape[0] // 4  # unrealistic that it'll fit if it takes up 25% of the memory
        if batches > max_size:
            for idx in range(0, batches, max_size):  # ... so we slice it up and feed in many smaller videos
                self.put(obj[idx:idx + max_size])
            return

        def _fits():
            return bool(self._free_memory(batches))

        # until new frames fit into memory
        waiting = 12
        while not _fits():
            time.sleep(5)
            waiting -= 1
        if not waiting:
            print("Warning: waited for one minute for space to free up, but none found. Increase memory size to avoid "
                  "fragmentation or implement defragmentation. Timestamp:", datetime.datetime.now(), flush=True)
            return

        self._put_item(obj)

    def __bool__(self):
        return bool(self.indices)


def frame_worker(work: list, worker_id: int, lock: threading.Semaphore, target_image_size: int, target_fps: int,
                 batch_size: int, queue_export):
    queue = SharedQueue.from_export(*queue_export)
    youtube_base = 'https://www.youtube.com/watch?v='
    youtube_getter = youtube_dl.YoutubeDL(
            {'writeautomaticsub': False, 'socket_timeout': 600, "quiet": True, "verbose": False, "no_warnings": True,
             "ignoreerrors": True
             })
    youtube_getter.add_default_info_extractors()
    random.Random(worker_id).shuffle(work)

    for wor in work:
        video_urls = get_video_urls(youtube_getter, youtube_base, wor, lock, target_image_size)

        if not video_urls:
            continue

        frames = get_video_frames(video_urls, target_image_size, target_fps)

        if frames is None or not frames.size:
            continue

        frames: np.ndarray = frames
        frames = frames[:frames.shape[0] // batch_size * batch_size]
        frames = frames.transpose((0, 3, 1, 2)).reshape((-1, batch_size, 3, target_image_size, target_image_size))
        queue.put(frames)


def main():
    workers, bucket, prefix, tmp_dir, urls, fps, batch_size, gpus, model_path, \
    shared_memory, chunk_size, video_downloaders = parse_args()
    config_path = f'{model_path}/vqgan.gumbelf8.config.yml'
    model_path = f'{model_path}/sber.gumbelf8.ckpt'
    if not os.path.exists(config_path):
        gdown.download(f'https://drive.google.com/uc?id=1WP6Li2Po8xYcQPGMpmaxIlI1yPB5lF5m', model_path, quiet=True)
    if not os.path.exists(config_path):
        gdown.download(f'https://drive.google.com/uc?id=1M7RvSoiuKBwpF-98sScKng0lsZnwFebR', config_path, quiet=True)
    os.makedirs(tmp_dir, exist_ok=True)
    conf = OmegaConf.load(config_path)
    padding_token = conf.model.params.n_embed
    resolution = conf.model.params.ddconfig.resolution
    model = load_vqgan(config_path, model_path)

    shared_memory = shared_memory * 1024 ** 3  # it's in GB, we have to convert it to bytes
    shared_frames = shared_memory // (256 ** 2 * 3 * batch_size)
    queue = SharedQueue.from_shape([shared_frames, batch_size, 3, 256, 256])

    ids = []
    for path in os.listdir(urls):
        with open(f'{urls}/{path}', 'rb') as f:
            video_ids, _ = pickle.load(f)
            ids.extend(video_ids)
    del video_ids

    ids = [ids[int(len(ids) * i / workers):int(len(ids) * (i + 1) / workers)] for i in range(workers)]
    lock = multiprocessing.Semaphore(video_downloaders)
    procs = [multiprocessing.Process(args=(work, worker_id, lock, resolution, fps, batch_size, queue.export()),
                                     daemon=True, target=frame_worker) for worker_id, work in enumerate(ids)]
    for p in procs:
        p.start()

    while not queue:  # "pre-wait" to get more accurate FPS counters
        time.sleep(1)

    bucket = boto3.resource("s3").Bucket(bucket)
    threads = [threading.Thread(target=worker,
                                args=(model, prefix, tmp_dir, bucket, i, queue, procs, chunk_size, padding_token),
                                daemon=True)
               for i in range(gpus)]

    for t in threads:
        t.start()

    for p in procs + threads:
        p.join()

    queue.frame_mem.unlink()
    queue.frame_mem.close()


if __name__ == '__main__':
    main()
