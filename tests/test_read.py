import os
import numpy as np
import pytest
import tempfile

from video2numpy import video2numpy


FRAME_COUNTS = {
    "vid1.mp4": 56,
    "vid2.mp4": 134,
    "https://www.youtube.com/watch?v=EKtBQbK4IX0": 78,
}


def test_read():
    test_path = "tests/test_videos"
    with tempfile.TemporaryDirectory() as tmpdir:
        video2numpy(os.path.join(test_path, "test_list.txt"), tmpdir, take_every_nth=2, resize_size=100)
        for vid in FRAME_COUNTS.keys():
            if vid.endswith(".mp4"):
                ld = vid[:-4] + ".npy"
            else:
                ld = vid.split("=")[-1] + ".npy"

            frames = np.load(os.path.join(tmpdir, ld))
            assert frames.shape[0] == FRAME_COUNTS[vid] // 2  # frame count
            assert frames.shape[1:] == (100, 100, 3)  # embed dim
