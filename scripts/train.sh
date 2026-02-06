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


source /software/rapids/r24.10/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatiallm
module load CUDA/12.4.0
cd $SLURM_SUBMIT_DIR

DATA_ROOT='/data/horse/ws/jixu233b-metadata_ws/datasets/arkitscenes-spatiallm/'

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=4  # Adjust to the number of GPUs available


DATA_ROOT='/data/horse/ws/jixu233b-metadata_ws/datasets/arkitscenes-spatiallm/'
DATA_ROOT='/mnt/nct-zfs/TCO-All/SharedDatasets/arkitscenes-spatiallm/'

# Experiment name (optional)
# Usage: EXPNAME="test" bash train.sh
# - If not set: output_dir will be {base_dir}/MMDDHHMM
# - If set: output_dir will be {base_dir}/{EXPNAME}_MMDDHHMM
# EXPNAME=${EXPNAME:-""}
#EXPNAME=${EXPNAME:-"spatiallm_cca_24_adaptedNorm"}
EXPNAME=${EXPNAME:-"spatiallm_cca_48_adaptedNorm"}
#EXPNAME=${EXPNAME:-"spatiallm_cca_24_gridsizeNorm"}

SPATIALLM_VERBOSE=0 python train.py \
    configs/spatiallm_sft_cca.yaml \
    expname="$EXPNAME"
