#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1 #2
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --gpus-per-task=1
#SBATCH --time=10:00:00
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

# cd /mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialLM
DATA_ROOT='/data/horse/ws/jixu233b-metadata_ws/datasets/arkitscenes-spatiallm/'
# DATA_ROOT='/mnt/nct-zfs/TCO-All/SharedDatasets/arkitscenes-spatiallm/'
scene_name='40753679'
scene_name='40753686'

# suggest not to disble, as sonata was trained with enable. Performance will slightly diff.
#--disable_flash_attn \
python inference.py \
--point_cloud ${DATA_ROOT}pcd/${scene_name}.ply \
--output outputs/scene${scene_name}.txt \
--model_path manycore-research/SpatialLM1.1-Qwen-0.5B \
--model_path ysmao/SpatialLM1.1-Qwen-0.5B-Structured3D-SFT \
--model_path ysmao/SpatialLM1.1-Qwen-0.5B-Arkitscenes-SFT \
--disable_flash_attn \
--disable_do_sample \
--VLM_PE CCA_2DProj \

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
