#!/bin/sh

# # # # # # # # # # # # # # # #
# SCRIPT TO ENQUEUE SLURM JOB
#
# PARAMETERS:
#       python_path: <path to python exec with requirements installed>"
#       config_file: <path to yaml configuration file>"
#
# DISCUSSION:
#       This script is prepared to run within the slurm folder
#       to keep the outputs of slurm in this folder. Hence the project variable.
#       Also, the python exec should also contain the tgnotify command to
#       notify via telegram when job has started/finished.
#       Please check: https://github.com/rusito-23/tg-notify to install
#       or comment the lines.
# # # # # # # # # # # # # # # #

PYTHON_PATH=$1
CONFIG_FILE=$2
CUDA_DEVICE=${3:-0}
CWD=$PWD
PROJECT_PATH=..
CUDA_VERSION=10.0
CUDNN_VERSION=v7.6-cu10.0

cd $PROJECT_PATH

tgnotify \
    --title "BackSeg" \
    --subtitle "Traning started"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/$CUDA_VERSION/extras/CUPTI/lib64/:/opt/cuda/$CUDA_VERSION/lib64:/opt/cudnn/$CUDNN_VERSION/
export CUDA_HOME=/opt/cuda/$CUDA_VERSION
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

$PYTHON_PATH train/train.py \
    --config-file $CONFIG_FILE

tgnotify \
    --title "BackSeg" \
    --subtitle "Traning finished"
  
cd $CWD
