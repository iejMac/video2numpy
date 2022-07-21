# video2numpy
[![pypi](https://img.shields.io/pypi/v/video2numpy.svg)](https://pypi.python.org/pypi/video2numpy)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iejMac/video2numpy/blob/master/notebook/video2numpy.ipynb)
[![Try it on gitpod](https://img.shields.io/badge/try-on%20gitpod-brightgreen.svg)](https://gitpod.io/#https://github.com/iejMac/video2numpy)

Optimized library for large-scale extraction of frames and audio from video.


## Install
```
pip install video2numpy
```

Or build from source:
```
python setup.py install
```

## Usage
```
NAME
    video2numpy - Read frames from videos and save as numpy arrays

SYNOPSIS
    video2numpy SRC <flags>

DESCRIPTION
    Input:
    src:
        str: path to mp4 file
        str: youtube link
        str: path to txt file with multiple mp4's or youtube links
        list: list with multiple mp4's or youtube links
    dest:
        str: directory where to save frames to
        None: dest = src + .npy
    take_every_nth:
        int: only take every nth frame
    resize_size:
        int: new pixel height and width of resized frame
    workers:
        int: number of workers used to read videos
    memory_size:
        int: number of GB of shared memory used for reading, use larger shared memory for more videos

POSITIONAL ARGUMENTS
    SRC

FLAGS
    --dest=DEST
        Default: ''
    --take_every_nth=TAKE_EVERY_NTH
        Default: 1
    --resize_size=RESIZE_SIZE
        Default: 224
    --workers=WORKERS
        Default: 1
    --memory_size=MEMORY_SIZE
        Default: 4

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```

## API

This module exposes a single function `video2numpy` which takes the same arguments as the command line tool:
```python
import glob
from video2numpy import video2numpy

VIDS = glob.glob("some/path/my_videos/*.mp4")
FRAME_DIR = "some/path/my_frames"
take_every_5 = 5

video2numpy(VIDS, FRAME_DIR, take_every_5)
```

You can also directly use the reader and iterate over videos yourself:
```python
import glob
from video2numpy.frame_reader import FrameReader

VIDS = glob.glob("some/path/my_videos/*.mp4")
take_every_5 = 5
resize_size = 300
batch_size = 64 # output shape will be (n, batch_size, height, width, 3)

reader = FrameReader(VIDS, take_every_5, resize_size, batch_size)
reader.start_reading()

for vid_frames, info_dict in reader:
    # info_dict["dst_name"] - name for saving numpy array
    # info_dict["pad_by"] - how many pad frames were added to final block so n_frames % batch_size == 0
    # do something with vid_frames of shape (n_blocks, 64, 300, 300, 3)
    ...
```

## For development

Either locally, or in [gitpod](https://gitpod.io/#https://github.com/rom1504/video2numpy) (do `export PIP_USER=false` there)

Setup a virtualenv:

```
python3 -m venv .env
source .env/bin/activate
pip install -e .
```

to run tests:
```
pip install -r requirements-test.txt
```
then 
```
make lint
make test
```

You can use `make black` to reformat the code

`python -m pytest -x -s -v tests -k "dummy"` to run a specific test
