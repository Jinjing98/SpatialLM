#!/bin/bash
# DATA_ROOT='/data/horse/ws/jixu233b-metadata_ws/datasets/arkitscenes-spatiallm/'
DATA_ROOT='/mnt/nct-zfs/TCO-All/SharedDatasets/arkitscenes-spatiallm/'
MAPPING_ROOT='/mnt/nct-zfs/TCO-All/SharedDatasets/SpatialLM-Testset/'
# JJ: Use trained SOPE checkpoint
MODEL_PATH='/mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/spatiallm/02141617/checkpoint-18/'
MODEL_PATH='/mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/spatiallm/02141740/checkpoint-18/'
MODEL_PATH='/mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/spatiallm/02141752/checkpoint-90/'
MODEL_PATH='ysmao/SpatialLM1.1-Qwen-0.5B-Arkitscenes-SFT'  # WARNING: This is mixedRoPE3D model!

# VLM_PE mode: SOPE (will auto-load config from checkpoint's config.yaml)
VLM_PE_MODE='sope'
VLM_PE_MODE='mixedRoPE3D'
VLM_PE_MODE='None'
VLM_PE_MODE='None'

OUTPUT_APPENDIX='_sope_ckpt18'
OUTPUT_APPENDIX='_mixedrope3d_ckpt90'
OUTPUT_APPENDIX='_none'
OUTPUT_APPENDIX='_baseline'

TEST_JSON_FILE='arkitscenes_val_small.json'
META_DATA_FILE='split_small.csv'
echo "=================================================="
echo "Evaluating ${MODEL_NAME} Model"
echo "Model: ${MODEL_PATH}"
echo "VLM_PE: ${VLM_PE_MODE}"
echo "Output: outputs/arkitscenes-spatiallm${OUTPUT_APPENDIX}/pred"
echo "=================================================="

# # Run inference on the PLY point clouds in folder SpatialLM-Testset/pcd with SOPE model
python inference.py \
--point_cloud ${DATA_ROOT}/pcd \
--output outputs/arkitscenes-spatiallm${OUTPUT_APPENDIX}/pred \
--model_path ${MODEL_PATH} \
--json_file ${DATA_ROOT}/${TEST_JSON_FILE} \
--VLM_PE ${VLM_PE_MODE} \
--disable_do_sample \

echo "=================================================="
echo "Inference complete! Starting evaluation..."
echo "=================================================="

# # Evaluate the predicted layouts
# python eval.py \
# --metadata ${DATA_ROOT}/${META_DATA_FILE} \
# --gt_dir ${DATA_ROOT}/layout \
# --pred_dir outputs/arkitscenes-spatiallm${OUTPUT_APPENDIX}/pred \
# --label_mapping ${MAPPING_ROOT}/benchmark_categories.tsv