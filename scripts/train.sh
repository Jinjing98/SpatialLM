#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1 #2
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --gpus-per-task=1
#SBATCH --time=30:00:00
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


export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=1  # Adjust to the number of GPUs available




# Experiment name (optional)
# Usage: EXPNAME="test" bash train.sh
# - If not set: output_dir will be {base_dir}/MMDDHHMM
# - If set: output_dir will be {base_dir}/{EXPNAME}_MMDDHHMM
# EXPNAME=${EXPNAME:-""}
#EXPNAME=${EXPNAME:-"spatiallm_cca_24_adaptedNorm"}
# EXPNAME=${EXPNAME:-"spatiallm_cca_48_adaptedNorm"}
# EXPNAME=${EXPNAME:-"spatiallm_cca_24_gridsizeNorm"}
# EXPNAME=${EXPNAME:-"spatiallm_sope_native_no_grad"}
EXPNAME=${EXPNAME:-"spatiallm_sope_native_no_grad_MAX_POINTS_PCD200k_MAX3072"}
# EXPNAME=${EXPNAME:-"spatiallm_axialrope3d_exp7_075_MAX_POINTS_fallbackthehack_detachPts_fixtypo"}
# EXPNAME=${EXPNAME:-"spatiallm_mixedrope3d_exp6_075_no_drift_leanredMixWeights_MAX_POINTS_PCD200k_MAX3072"}
# EXPNAME=${EXPNAME:-"spatiallm_mixedrope3d_exp5_075_no_drift_theta100_MAX_POINTS_PCD200k_MAX3072"}
# EXPNAME=${EXPNAME:-"spatiallm_mixedrope3d_exp4_075_no_drift_no_interleave_MAX_POINTS_PCD200k_MAX3072"}
# EXPNAME=${EXPNAME:-"spatiallm_mixedrope3d_exp3_025_no_drift_MAX_POINTS_PCD200k_MAX3072"}
# EXPNAME=${EXPNAME:-"spatiallm_mixedrope3d_exp3_025_no_drift_detachpts_MAX_POINTS_PCD200k_MAX3072"}
# EXPNAME=${EXPNAME:-"spatiallm_mixedrope3d_exp2_075_avg_drift_MAX_POINTS_PCD200k_MAX3072"}
# EXPNAME=${EXPNAME:-"spatiallm_mixedrope3d_exp1_075_no_drift_MAX_POINTS_PCD200k_MAX3072"}
# EXPNAME=${EXPNAME:-"spatiallm_dft1DRoPE_MAX_POINTS_PCD200k_MAX3072"}

# PYTORCH_MAX_POINTS_PCD200k_MAX3072=expandable_segments:True \
SPATIALLM_VERBOSE=0 python train.py \
    configs/spatiallm_sft_sope.yaml \
    expname="$EXPNAME"

    # configs/spatiallm_sft_mixedrope3d_exp6.yaml \
    # configs/spatiallm_sft_axialrope3d_exp7.yaml \
    # configs/spatiallm_sft_sope.yaml \
    # configs/spatiallm_sft.yaml \

