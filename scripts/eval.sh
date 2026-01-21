#!/bin/bash
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
