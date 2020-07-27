# ENQUEUE THE SLURM JOB USING CURRENT DIR AND PYTHON
# CHECK slurm DIR FOR MORE INFO 
# 
# PARAMETERS:
#       $1 : config_path: <path to yaml configuration file>
#       $2 : cuda device: <which cuda should be used (default 0)>

sbatch -o slurm/%a.out slurm/enqueue.sh $(pwd) $(which python) $1 $2
