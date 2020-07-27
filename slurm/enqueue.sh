#!/bin/bash
### -------------------------------------------
###             Enqueue Slurm Job
### Discussion:
###     Moves to project `slurm` path and executes the run file. 
###     This avoids creation of .out files outside the `slurm` folder
###     and uses the same logs folder for all jobs without messing the root project folder.
### Parameters:
###     - $1: PROJECT_PATH
###     - $2: PYTHON_PATH
###     - $3: CONFIG_FILE
###     - $4: CUDA_DEVICE (default=0)
### -------------------------------------------

### -------------------------------------------
### Configure job resources
### -------------------------------------------

### Job queue to use (options: batch)
#SBATCH --partition=batch

### Amount of nodes to use
#SBATCH --nodes=1

### Processes per node
#SBATCH --ntasks-per-node=1

### Available cores per node (1-12)
#SBATCH --cpus-per-task=1

### execution time. Format: days-hours:minutes:seconds -- Max: three days
#SBATCH --time 1-00:00:00

## ## Load environment modules
## module load compilers/gcc/4.9

### -------------------------------------------
### Configure variables
### -------------------------------------------

CWD=$PWD
PROJECT_PATH=$1
PYTHON_PATH=$2
CONFIG_FILE=$3
CUDA_DEVICE=$4

### move to project slurm path
cd $PROJECT_PATH/slurm

### run job
srun \
    -o logs/%j.out \
    -e logs/%j.err \
    /bin/bash run.sh $PYTHON_PATH $CONFIG_FILE $CUDA_DEVICE


### move back to curr dir
cd $CWD

exit 0
