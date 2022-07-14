#!/bin/bash

wget https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_001.tar.gz
tar -xvf k700_train_001.tar.gz
rm k700_train_001.tar.gz
mv abseiling benchmark_vids
