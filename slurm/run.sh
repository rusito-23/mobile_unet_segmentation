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

cd $PROJECT_PATH

tgnotify \
    --title "BackSeg" \
    --subtitle "Traning started"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/9.0/extras/CUPTI/lib64/:/opt/cuda/9.0/lib64:/opt/cudnn/v7.0/
export CUDA_HOME=/opt/cuda/9.0
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

$PYTHON_PATH mobile_unet_seg/train.py \
    --config-file $CONFIG_FILE

tgnotify \
    --title "BackSeg" \
    --subtitle "Traning finished"
  
cd $CWD
