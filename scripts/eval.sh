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

DATA_ROOT='/data/horse/ws/jixu233b-metadata_ws/datasets/arkitscenes-spatiallm/'


# Run inference on the PLY point clouds in folder SpatialLM-Testset/pcd with SpatialLM1.1-Qwen-0.5B model
python inference.py \
--point_cloud /data/horse/ws/jixu233b-metadata_ws/datasets/structured3d-spatiallm/pcd \
--output outputs/structured3d-spatiallm/pred \
--model_path ysmao/SpatialLM1.1-Qwen-0.5B-Structured3D-SFT \
--json_file /data/horse/ws/jixu233b-metadata_ws/datasets/structured3d-spatiallm/structured3d_test.json

# # Evaluate the predicted layouts
python eval.py \
--metadata /data/horse/ws/jixu233b-metadata_ws/datasets/structured3d-spatiallm/split.csv \
--gt_dir /data/horse/ws/jixu233b-metadata_ws/datasets/structured3d-spatiallm/layout \
--pred_dir outputs/structured3d-spatiallm/pred \

# --label_mapping /data/horse/ws/jixu233b-metadata_ws/datasets/structured3d-spatiallm/split.csv
