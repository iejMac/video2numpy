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

    vid_chunk_size = 1
    take_every_nth = 1
    resize_size = 150

    reader = FrameReader(vids, vid_chunk_size, take_every_nth, resize_size)

    reader.start_reading()

    for block, ind_dict in reader:
        for dst_name, inds in ind_dict.items():
            i0, it = inds
            frames = block[i0:it]

            mp4_name = dst_name[:-4] + ".mp4"

            assert it - i0 == FRAME_COUNTS[mp4_name]
            assert block.shape[0] == it - i0
            assert block.shape[1:] == (150, 150, 3)
    assert len(reader.shms) == 0


def test_resizer():
    fake_img = np.zeros((480, 640, 3))
    resizer = Resizer(from_shape=[480, 640, 3], to_size=100)

    resized_img = resizer(fake_img)
    assert resized_img.shape == (100, 100, 3)
