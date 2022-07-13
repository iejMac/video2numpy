import glob

from video2numpy.frame_reader import FrameReader


if __name__ == "__main__":
    vids = glob.glob("tests/test_videos/*.mp4")  # use test videos for demo

    chunk_size = 1
    take_every_nth = 10
    resize_size = 100

    # intialize reader with auto_release=False
    reader = FrameReader(vids, chunk_size, take_every_nth, resize_size, auto_release=False)
    reader.start_reading()

    blocks = []
    for block, ind_dict in reader:
        blocks.append(block)

    # will throw Segmentation fault if auto_relase=True (try it)
    for block in blocks:
        print(block[0, 0, 0, 0])  # 5, 63

    # Remember to release memory
    reader.release_memory()
