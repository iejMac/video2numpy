import cv2

from video2numpy.reader import FrameReader
from video2numpy.utils import split_block

if __name__ == "__main__":

    links = [  # random short videos
        "https://www.youtube.com/watch?v=14d_Kk77gg0",
        "https://www.youtube.com/watch?v=MHZl-LnpGRg",
        "https://www.youtube.com/watch?v=S6hwS-NB8c8",
        "https://www.youtube.com/watch?v=4RyoplkTxSg",
        "https://www.youtube.com/watch?v=j6kb5nHMV5A",
        "https://www.youtube.com/watch?v=pdyRT_BXfXE",
    ]

    chunk_size = 2  # two video per thread
    take_every_nth = 25  # take every 25th frame
    resize_size = 224  # make frames 224x224

    reader = FrameReader(links, chunk_size, take_every_nth, resize_size)
    reader.start_reading()

    for block, ind_dict in reader:

        vid_frames = split_block(block, ind_dict)

        # Play few frames from each video
        for vidID, frames in vid_frames.items():
            print(f"Playing video {vidID}...")

            for frame in frames:
                cv2.imshow("frame", frame)

                key = cv2.waitKey(50)
                if key == ord("q"):
                    break

    cv2.destroyAllWindows()
