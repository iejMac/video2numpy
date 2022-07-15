import time
import glob
import multiprocessing
import numpy as np

from shared_queue import SharedQueue
from read_vids_sq_cv2 import read_vids


if __name__ == "__main__":
    vids = glob.glob("../benchmark/benchmark_vids/*.mp4")
    vids = vids[:50]

    take_en = 1
    resize_size = 224
    workers = 6

    shared_mem = 4 * 1024 ** 3
    shared_frames = shared_mem// (256 ** 2 * 3)

    queue = SharedQueue.from_shape([shared_frames, resize_size, resize_size, 3])
  
    ids = vids
    ids = [ids[int(len(ids) * i / workers):int(len(ids) * (i + 1) / workers)] for i in range(workers)]

    procs = [multiprocessing.Process(args=(work, worker_id, take_en, resize_size, queue.export()), daemon=True, target=read_vids) for worker_id, work in enumerate(ids)]

    for p in procs:
        p.start()

    t0 = time.perf_counter()

    ct = 0
    for vid in vids:
        x = queue.get()
        print(x.shape)
        ct += x.shape[0]

    tot_time = time.perf_counter() - t0
    print(f"FPS: {ct/tot_time}")

    for p in procs:
        p.join()

    queue.frame_mem.unlink()
    queue.frame_mem.close()
