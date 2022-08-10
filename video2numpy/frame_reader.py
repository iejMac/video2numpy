"""reader - uses a reader function to read frames from videos"""
import multiprocessing
import psutil

import numpy as np

from .read_vids_cv2 import read_vids
from .shared_queue import SharedQueue

# NOTE: from experimentation this seems to be the best configuration
# TODO: specify hardware where this is best, test other hardware
CPUS_CHUNK = 6 # 6 workers use 6 cpus


class FrameReader:
    """
    Iterates over frame blocks returned by read_vids function
    """

    def __init__(
        self,
        vids,
        take_every_nth=1,
        resize_size=224,
        batch_size=-1,
        cpus=-1,
        memory_size=4,
    ):
        """
        Input:
          vids - list with youtube links or paths to mp4 files.
          chunk_size - how many videos to process at once.
          take_every_nth - offset between frames we take.
          resize_size - pixel height and width of target output shape.
          batch_size - max length of frame sequence to put on shared_queue (-1 = no max).
          cpus -
            int - number of cpus to use (-1 means all)
            list - cpu indices to use
          memory_size - number of GB of shared_memory
        """

        memory_size_b = int(memory_size * 1024**3)  # GB -> bytes
        shared_blocks = memory_size_b // (resize_size**2 * 3 * (1 if batch_size == -1 else batch_size))
        dim12 = (shared_blocks,) if batch_size == -1 else (shared_blocks, batch_size)
        self.shared_queue = SharedQueue.from_shape([*dim12, resize_size, resize_size, 3])

        # NOTE: might not work on SLURM-type compute nodes where you can have htop show 48 cores but you only
        # have access to a fraction of them (this will always return all 48)
        p = psutil.Process()
        cpu_avail = p.cpu_affinity()
        if type(cpus) == int:
            cpu_avail = cpu_avail[:cpus] if cpus != -1 else cpu_avail
        elif type(cpus) == list:
            cpu_avail = cpus

        cpu_ct = len(cpu_avail)
        workers = cpu_ct

        div_vids = [s.tolist() for s in np.array_split(vids, workers)]
        resource_groups = [cpu_avail[i*CPUS_CHUNK:(i+1)*CPUS_CHUNK] for i in range(cpu_ct//CPUS_CHUNK + (cpu_ct % CPUS_CHUNK != 0))]

        div_resources = [resource_groups[i//CPUS_CHUNK] for i in range(workers)]

        print(div_resources)

        work_res = zip(div_vids, div_resources)

        self.procs = [
            multiprocessing.Process(
                args=(work, worker_id, res, take_every_nth, resize_size, batch_size, self.shared_queue.export()),
                daemon=True,
                target=read_vids,
            )
            for worker_id, (work, res) in enumerate(work_res)
        ]

    def __iter__(self):
        return self

    def __next__(self):
        if self.shared_queue or any(p.is_alive() for p in self.procs):
            frames, info = self.shared_queue.get()
            return frames, info
        self.finish_reading()
        self.release_memory()
        raise StopIteration

    def start_reading(self):
        for p in self.procs:
            p.start()

    def finish_reading(self):
        for p in self.procs:
            p.join()

    def release_memory(self):
        self.shared_queue.frame_mem.unlink()
        self.shared_queue.frame_mem.close()
