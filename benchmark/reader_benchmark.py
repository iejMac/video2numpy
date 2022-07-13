import glob
import time

from video2numpy.frame_reader import FrameReader


TRUE_CPU_COUNT = 6 # TODO find a way to get this automatically
CONST_VID_FPS = 25


def benchmark_reading(vids, chunk_size, take_en, resize_size, thread_count):
    reader = FrameReader(
        vids,
        chunk_size=chunk_size,
        take_every_nth=take_en,
        resize_size=resize_size,
        thread_count=thread_count,
        auto_release=True,
    )
    reader.start_reading() 

    t0 = time.perf_counter()

    count = 0
    for block, ind_dict in reader:
        block[0,0,0,0] # assert no Segmentation fault
        count += block.shape[0]

    read_time = time.perf_counter() - t0
    samp_per_s = count / read_time
    return samp_per_s, count, read_time


if __name__ == "__main__":
    vids = glob.glob("benchmark/benchmark_vids/*.mp4")
    vids = vids[:100]  # TODO: remove this

    print(f"Benchmarking on {len(vids)} videos...")

    video_fps = [25, 15, 10, 5, 1] # tested variable
    chunk_size = 50
    resize_size = 224
    thread_count = TRUE_CPU_COUNT

    results = []
    for fps in video_fps:
        ten = int(CONST_VID_FPS/fps)
        samp_per_s, _, _ = benchmark_reading(
            vids,
            chunk_size,
            ten,
            resize_size,
            thread_count
        )
        results.append((fps, samp_per_s))
