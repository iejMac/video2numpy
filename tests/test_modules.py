import glob

import numpy as np

from video2numpy.frame_reader import FrameReader
from video2numpy.resizer import Resizer


FRAME_COUNTS = {
    "vid1.mp4": 56,
    "vid2.mp4": 134,
}


def test_reader():
    vids = glob.glob("tests/test_videos/*.mp4")

    take_every_nth = 1
    resize_size = 150
    batch_size = 5

    reader = FrameReader(vids, None, take_every_nth, resize_size, batch_size, memory_size=0.128)
    reader.start_reading()

    for vid_frames, info in reader:
        vid_frames[0, 0, 0, 0, 0]  # assert still allocated
        mp4_name = info["dst_name"][:-4] + ".mp4"
        frame_count = vid_frames.shape[0] * vid_frames.shape[1] - info["pad_by"]
        assert frame_count == FRAME_COUNTS[mp4_name]


def test_resizer():
    fake_img = np.zeros((480, 640, 3))
    resizer = Resizer(from_shape=[480, 640, 3], to_size=100)

    resized_img = resizer(fake_img)
    assert resized_img.shape == (100, 100, 3)
