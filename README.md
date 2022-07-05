# video2numpy
[![pypi](https://img.shields.io/pypi/v/video2numpy.svg)](https://pypi.python.org/pypi/video2numpy)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rom1504/video2numpy/blob/master/notebook/video2numpy.ipynb)
[![Try it on gitpod](https://img.shields.io/badge/try-on%20gitpod-brightgreen.svg)](https://gitpod.io/#https://github.com/rom1504/video2numpy)

A nice template to start with

## Install

pip install video2numpy

## API

This module exposes a single function `hello_world` which takes the same arguments as the command line tool:

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
