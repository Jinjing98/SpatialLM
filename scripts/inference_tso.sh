#!/bin/bash
# cd /mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialLM
DATA_ROOT='/data/horse/ws/jixu233b-metadata_ws/datasets/arkitscenes-spatiallm/'
DATA_ROOT='/mnt/nct-zfs/TCO-All/SharedDatasets/arkitscenes-spatiallm/'
scene_name='40753679'
#scene_name='40753686'

# Model path options:
# 1. For mixedRoPE3D inference (trained model):
#    MODEL_PATH="ysmao/SpatialLM1.1-Qwen-0.5B-Arkitscenes-SFT"
# 2. For SOPE inference (use your trained checkpoint):
#    MODEL_PATH="/mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/spatiallm/spatiallm_YYYYMMDD_HHMMSS/checkpoint-XXXX"
# 3. For rough SOPE testing with base model (results may not be meaningful):
#    MODEL_PATH="manycore-research/SpatialLM1.1-Qwen-0.5B"

# Choose VLM_PE mode:
# VLM_PE_MODE="sope"          # Options: "mixedRoPE3D", "sope"
VLM_PE_MODE="mixedRoPE3D"          # Options: "mixedRoPE3D", "sope"
MODEL_PATH="ysmao/SpatialLM1.1-Qwen-0.5B-Arkitscenes-SFT"  # WARNING: This is trained with mixedRoPE3D!

# suggest not to disble, as sonata was trained with enable. Performance will slightly diff.
#--disable_flash_attn \
python inference.py \
--point_cloud ${DATA_ROOT}pcd/${scene_name}.ply \
--output outputs/scene${scene_name}.txt \
--model_path ${MODEL_PATH} \
--disable_do_sample \
--disable_flash_attn \
--VLM_PE ${VLM_PE_MODE} \
# # # Convert the predicted layout to Rerun format
# run below on my local 
python visualize.py \
--point_cloud ${DATA_ROOT}pcd/${scene_name}.ply \
--layout outputs/scene${scene_name}.txt \
--save outputs/scene${scene_name}.rrd

# vis result on local machine : workflow
# (base) jinjingxu@G27LP0076-Linux:~$ pip install --upgrade rerun-sdk
# (optional)
# (base) jinjingxu@G27LP0076-Linux:~$ cp /mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialLM/outputs/scene${scene_name}.rrd /tmp/test.rrd
# (base) jinjingxu@G27LP0076-Linux:~$ rerun /tmp/test.rrd 
