#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=hface_prepro
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/sbatch.log

# REMEMBER TO CHANGE: --mem, --gres, --gpus-per-node, --time
echo "Inside sbatch_run.sh script..."

module purge
deactivate
# module load PyTorch
# source ~/group_space/robin/envs/hugface/bin/activate


DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

x=`pwd`
WORKDIR=${x##*/}
TARGET_DIR="/workspace/$WORKDIR"
CONTAINER_PATH="/ceph/hpc/home/eurobink/group_space/containers/megatron-deepspeed.sif"
FILE=$1
FILENAME=${FILE##*/}


PROJECT=/ceph/hpc/home/eurobink/group_space/robin/workspace/$WORKDIR
LOGGING=$PROJECT/logs
export HF_DATASETS_CACHE=$x/hface_datasets_cache

# run_cmd="python prepare_data.py robin_args.json"
run_cmd="bash run_prepro.sh $FILE"
# $run_cmd
srun -l --output=$LOGGING/prepro_"$DATETIME"_$FILENAME.log \
    singularity exec --nv --pwd /workspace/$WORKDIR --bind $PROJECT:$TARGET_DIR $CONTAINER_PATH $run_cmd
set +x

exit 0