#!/bin/bash

#SBATCH --job-name=end_pose
#SBATCH --gpus=v100:1   # rtxa5000 p6000 rtx6000 a100 v100 # monst3r requires 48GB each, only a100 supports
#SBATCH --nodes=1  # several gpus on one node
#SBATCH --ntasks-per-node=1 #used for multi gpu training
#SBATCH --mem=48G #64G #35G#25G  # 20G may cause bus error?   # mem * num_GPUS
#SBATCH --time=46:00:00
#SBATCH --cpus-per-task=4 #8 #4   #num works4 can not be too big;
#SBATCH --mail-user=xu.jinjing@uniklinikum-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/mnt/nct-zfs/TCO-Test/jinjingxu/slurm_out/%j.err
#SBATCH --output=/mnt/nct-zfs/TCO-Test/jinjingxu/slurm_out/%j.out


DATA_ROOT='/data/horse/ws/jixu233b-metadata_ws/datasets/arkitscenes-spatiallm/'

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=4  # Adjust to the number of GPUs available


DATA_ROOT='/data/horse/ws/jixu233b-metadata_ws/datasets/arkitscenes-spatiallm/'
DATA_ROOT='/mnt/nct-zfs/TCO-All/SharedDatasets/arkitscenes-spatiallm/'

# suggest not to disble, as sonata was trained with enable. Performance will slightly diff.
# Note: output_dir will automatically have timestamp appended (e.g., saves_0126_153045)
# To disable: AUTO_TIMESTAMP_OUTPUT_DIR=0 python train.py ...
SPATIALLM_VERBOSE=0 python train.py \
configs/spatiallm_sft.yaml

# Override config parameters from command line (examples):
# SPATIALLM_VERBOSE=0 python train.py configs/spatiallm_sft.yaml \
#   vlm_pe="3D_RoPE" \
#   pcd_pe_merge_rule="3D_with_1D" \
#   pcd_theta=10000 \
#   disable_flash_attn=false
#
# Or use different PE methods:
# vlm_pe="3D_Sinusoidal" pcd_pe_merge_rule="3D_only"
# vlm_pe="None"  # Use standard 1D RoPE (original SpatialLM behavior)
#
# You can also set these directly in configs/spatiallm_sft.yaml

# # Run inference on the PLY point clouds in folder SpatialLM-Testset/pcd with SpatialLM1.1-Qwen-0.5B model
# python inference.py \
# --point_cloud /data/horse/ws/jixu233b-metadata_ws/datasets/structured3d-spatiallm/pcd \
# --output outputs/structured3d-spatiallm/pred \
# --model_path ysmao/SpatialLM1.1-Qwen-0.5B-Structured3D-SFT \
# --json_file /data/horse/ws/jixu233b-metadata_ws/datasets/structured3d-spatiallm/structured3d_test.json

# # # Evaluate the predicted layouts
# python eval.py \
# --metadata /data/horse/ws/jixu233b-metadata_ws/datasets/structured3d-spatiallm/split.csv \
# --gt_dir /data/horse/ws/jixu233b-metadata_ws/datasets/structured3d-spatiallm/layout \
# --pred_dir outputs/structured3d-spatiallm/pred \

# # --label_mapping /data/horse/ws/jixu233b-metadata_ws/datasets/structured3d-spatiallm/split.csv
