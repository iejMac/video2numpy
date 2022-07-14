import argparse
import glob
import time

from matplotlib import pyplot as plt

from video2numpy.frame_reader import FrameReader


CONST_VID_FPS = 25
BENCH_DATASET_SIZE = 300


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        default="default",  # TODO: maybe find nice way of getting reading type (cv2)
        help="For unique output graph file name",
    )
    parser.add_argument("--chunk_size", type=int, default=50, help="How many videos to try to read at once")
    parser.add_argument("--resize_size", type=int, default=224, help="Resize frames to resize_size x resize_size")
    parser.add_argument(
        "--thread_count",
        type=int,
        default=6,  # TODO: find way of getting default automatically
        help="How many threads to use for video chunk reading",
    )
    args = parser.parse_args()
    return args


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
        block[0, 0, 0, 0]  # assert no Segmentation fault
        count += block.shape[0]

    read_time = time.perf_counter() - t0
    samp_per_s = count / read_time
    return samp_per_s, count, read_time


if __name__ == "__main__":
    args = parse_args()
    vids = glob.glob("benchmark/benchmark_vids/*.mp4")
    vids = vids[:BENCH_DATASET_SIZE]

    print(f"Benchmarking {args.name} on {len(vids)} videos...")

    video_fps = [1, 3, 5, 10, 25]  # tested variable
    chunk_size = args.chunk_size
    resize_size = args.resize_size
    thread_count = args.thread_count

    print(f"Chunk size - {chunk_size} | Resize size - {resize_size} | Thread count - {thread_count}")

    results = []
    for fps in video_fps:
        ten = int(CONST_VID_FPS / fps)
        samp_per_s, _, _ = benchmark_reading(vids, chunk_size, ten, resize_size, thread_count)
        results.append(samp_per_s)

    plt.plot(video_fps, results)
    plt.title(f"{args.name}: chunk size - {chunk_size} | resize size - {resize_size} | threads - {thread_count}")
    plt.xlabel("Target video FPS")
    plt.ylabel("Reading speed samples/s")
    plt.savefig(f"eff_{args.name}_{chunk_size}_{resize_size}_{thread_count}.png")
