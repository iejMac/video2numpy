## 2.3.0

* FrameReader can take a list of references to videos for linking to related data on the other end of video2numpy (f.e. index in csv or parquet)
* More output from video2numpy about what's going on during processing.

## 2.2.0

* FrameReader has __len__ implemented as number of videos
* support for .mp4 links
* small bug fixes in SharedQueue

## 2.1.0

* FrameReader has the option to output batched blocks of frames

## 2.0.0

* Use single SharedQueue to store frames
* FrameReader iterates over video frames instead of frame blocks
* Fixed shared memory errors

## 1.1.0

* Allow user to specify thread count

## 1.0.1

* fix FileNotFoundError when unlinking shared memory

## 1.0.0

* it works
