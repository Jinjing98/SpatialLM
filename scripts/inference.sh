#!/bin/bash
# cd /mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialLM
DATA_ROOT='/data/horse/ws/jixu233b-metadata_ws/datasets/arkitscenes-spatiallm/'
DATA_ROOT='/mnt/nct-zfs/TCO-All/SharedDatasets/arkitscenes-spatiallm/'

python inference.py \
--point_cloud ${DATA_ROOT}pcd/40753679.ply \
--output outputs/scene40753679.txt \
--model_path manycore-research/SpatialLM1.1-Qwen-0.5B \
--model_path ysmao/SpatialLM1.1-Qwen-0.5B-Structured3D-SFT \

# # Convert the predicted layout to Rerun format
python visualize.py \
--point_cloud ${DATA_ROOT}pcd/40753679.ply \
--layout outputs/scene40753679.txt \
--save outputs/scene40753679.rrd