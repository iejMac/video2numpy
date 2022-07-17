import cv2

from video2numpy.frame_reader import FrameReader

if __name__ == "__main__":

    links = [  # random short videos
        "https://www.youtube.com/watch?v=14d_Kk77gg0",
        "https://www.youtube.com/watch?v=MHZl-LnpGRg",
        "https://www.youtube.com/watch?v=S6hwS-NB8c8",
        "https://www.youtube.com/watch?v=4RyoplkTxSg",
        "https://www.youtube.com/watch?v=j6kb5nHMV5A",
        "https://www.youtube.com/watch?v=pdyRT_BXfXE",
    ]

    take_every_nth = 25  # take every 25th frame
    resize_size = 224  # make frames 224x224

    reader = FrameReader(links, take_every_nth, resize_size)
    reader.start_reading()

    for frames, vidID in reader:
        # Play few frames from each video
        print(f"Playing video {vidID}...")
        for frame in frames:
            cv2.imshow("frame", frame)

            key = cv2.waitKey(50)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
