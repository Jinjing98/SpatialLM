#!/bin/bash
# cd /mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialLM
DATA_ROOT='/data/horse/ws/jixu233b-metadata_ws/datasets/arkitscenes-spatiallm/'
DATA_ROOT='/mnt/nct-zfs/TCO-All/SharedDatasets/arkitscenes-spatiallm/'

# suggest not to disble, as sonata was trained with enable. Performance will slightly diff.
#--disable_flash_attn \
SPATIALLM_VERBOSE=0 python inference.py \
--point_cloud ${DATA_ROOT}pcd/40753679.ply \
--output outputs/scene40753679.txt \
--model_path manycore-research/SpatialLM1.1-Qwen-0.5B \
--model_path ysmao/SpatialLM1.1-Qwen-0.5B-Structured3D-SFT \
--model_path ysmao/SpatialLM1.1-Qwen-0.5B-Arkitscenes-SFT \
--vlm_pe "3D_RoPE" \
--pcd_pe_merge_rule "3D_with_1D" \
# --vlm_pe "3D_Sinusoidal" \
# --disable_flash_attn \
# --vlm_pe "CCA_2DProj" \

# # # Convert the predicted layout to Rerun format
# run below on my local 
python visualize.py \
--point_cloud ${DATA_ROOT}pcd/40753679.ply \
--layout outputs/scene40753679.txt \
--save outputs/scene40753679.rrd

# vis result on local machine : workflow
# (base) jinjingxu@G27LP0076-Linux:~$ pip install --upgrade rerun-sdk
# (optional)
# (base) jinjingxu@G27LP0076-Linux:~$ cp /mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialLM/outputs/scene40753679.rrd /tmp/test.rrd
# can also darg file to vis more .rrd
# (base) jinjingxu@G27LP0076-Linux:~$ rerun /tmp/test.rrd 
