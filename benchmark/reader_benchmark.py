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
    parser.add_argument(
        "--resize_size",
        type=int,
        default=224,
        help="Resize frames to resize_size x resize_size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,  # TODO: find way of getting default automatically
        help="How many workers to use for video reading",
    )
    args = parser.parse_args()
    return args


def benchmark_reading(vids, take_en, resize_size, workers):
    reader = FrameReader(
        vids,
        take_every_nth=take_en,
        resize_size=resize_size,
        workers=workers,
        memory_size=4,
    )
    reader.start_reading()

    t0 = time.perf_counter()

    count = 0
    for vid, name in reader:
        vid[0, 0, 0, 0]  # assert no Segmentation fault
        count += vid.shape[0]

    read_time = time.perf_counter() - t0
    samp_per_s = count / read_time
    return samp_per_s, count, read_time


if __name__ == "__main__":
    args = parse_args()
    vids = glob.glob("benchmark/benchmark_vids/*.mp4")
    vids = vids[:BENCH_DATASET_SIZE]

    print(f"Benchmarking {args.name} on {len(vids)} videos...")

    video_fps = [1, 3, 5, 10, 25]  # tested variable
    resize_size = args.resize_size
    workers = args.workers

    print(f"Resize size - {resize_size} | Workers - {workers}")

    results = []
    for fps in video_fps:
        ten = int(CONST_VID_FPS / fps)
        samp_per_s, _, _ = benchmark_reading(vids, ten, resize_size, workers)
        print(f"samples/s @ {fps} FPS = {samp_per_s}")
        results.append(samp_per_s)
        time.sleep(5)  # allow time for reset

    plt.plot(video_fps, results)
    plt.title(f"{args.name}: resize size - {resize_size} | workers - {workers}")
    plt.xlabel("Target video FPS")
    plt.ylabel("Reading speed samples/s")
    plt.savefig(f"eff_{args.name}_{workers}_{resize_size}_{workers}.png")
