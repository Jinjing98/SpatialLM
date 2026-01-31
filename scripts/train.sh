#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1 #2
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --gpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=80G
#SBATCH --partition=capella
#SBATCH --mail-user=xvjinjing8@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.err
#SBATCH --output=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.out

# load modules
# module load cuda/11.8
# module load python/3.10

# activate conda
source /software/rapids/r24.10/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatiallm

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=4  # Adjust to the number of GPUs available


DATA_ROOT='/data/horse/ws/jixu233b-metadata_ws/datasets/arkitscenes-spatiallm/'
SPATIALLM_VERBOSE=0 python train.py \
configs/spatiallm_sft.yaml
 