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

POSITIONAL ARGUMENTS
    SRC

FLAGS
    --dest=DEST
        Default: ''
    --take_every_nth=TAKE_EVERY_NTH
        Default: 1
    --resize_size=RESIZE_SIZE
        Default: 224
```

## API

This module exposes a single function `video2numpy` which takes the same arguments as the command line tool:
```python
import glob
from video2numpy import video2numpy

VIDS = glob.glob("some/path/my_videos/*.mp4")
FRAME_DIR = "some/path/my_embeddings"
take_every_5 = 5

video2numpy(VIDS, FRAME_DIR, take_every_5)
```

You can alse directly use the reader and iterate over frame blocks yourself:
```python
import glob
from video2numpy.frame_reader import FrameReader
from video2numpy.utils import split_block

VIDS = glob.glob("some/path/my_videos/*.mp4")
take_every_5 = 5
resize_size = 300

reader = FrameReader(VIDS, FRAME_DIR, take_every_5, resize_size)

for block, ind_dict in reader:

    if you need to process the block in large batches (f.e. good for ML):
        proc_block = ml_model(block)
    else:
        proc_block = block

    # then you can separate the video frames into a dict easily with split_block from utils:
    split_up_vids = split_block(proc_block, ind_dict)

    for vid_name, proc_frames in split_up_vids.items():
        # do something with proc_frame of shape (n_frames, 300, 300, 3)
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
