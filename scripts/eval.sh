#!/bin/bash
cd /mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialLM

# Run inference on the PLY point clouds in folder SpatialLM-Testset/pcd with SpatialLM1.1-Qwen-0.5B model
python inference.py \
--point_cloud /mnt/nct-zfs/TCO-All/SharedDatasets/structured3d-spatiallm/pcd \
--output outputs/structured3d-spatiallm/pred \
--model_path ysmao/SpatialLM1.1-Qwen-0.5B-Structured3D-SFT \
--json_file /mnt/nct-zfs/TCO-All/SharedDatasets/structured3d-spatiallm/structured3d_test.json

# # Evaluate the predicted layouts
# python eval.py \
# --metadata /mnt/nct-zfs/TCO-All/SharedDatasets/structured3d-spatiallm/split.csv
# --gt_dir /mnt/nct-zfs/TCO-All/SharedDatasets/structured3d-spatiallm/layout \
# --pred_dir outputs/structured3d-spatiallm/pred \

# --label_mapping /mnt/nct-zfs/TCO-All/SharedDatasets/structured3d-spatiallm/split.csv
