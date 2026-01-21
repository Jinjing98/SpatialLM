# set up the spatiallm env
# pip install --no-build-isolation git+https://github.com/mit-han-lab/torchsparse.git
# Go to your third_party directory
# cd SpatialLM/third_party/sonata
# pip install -e .
# Go to your third_party directory
# cd /mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialLM/third_party
# # Clone spconv if not already there
# git clone https://github.com/traveller59/spconv.git
# cd spconv
# git checkout v2.2.0  # compatible with PyTorch 2.2
# # Build and install into your environment in editable mode
# python -m pip install -e .
# Recommended suite for 3D/VLM spatial encoders
#pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
# Install the pre-compiled wheel for your specific setup
#pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.2cxx11abiFalse-cp311-cp311-linux_x86_64.whl

# Install ipykernel (SO THAT CAN BE DETECTED IN NOTEBOOK)
# conda install ipykernel -c conda-forge
# python -m ipykernel install --user --name spatiallm --display-name "Python (SpatialLM)"

# #/////////////////////////
# #module load release/25.06
# #module load Anaconda3/2025.06-1

# #/////////////////////////
module purge
# # 1. Ensure conda is initialized in THIS shell
source /software/rapids/r24.10/Anaconda3/2024.02-1/etc/profile.d/conda.sh
# # 2. Hard deactivate anything fake
# conda deactivate || true
# # 3. Activate explicitly
conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatiallm
module load CUDA/12.4.0

#export CUMM_CUDA_ARCH_LIST="8.0" 
# 然后再次尝试进入 Python 导入
#python -c "import spconv; print(spconv.__version__)"

# (spatiallm) [jixu233b@login2.capella SpatialLM]$ srun --gres=gpu:1 -N 1 bash /home/jixu233b/Project
# s/VLM_3D/SpatialLM/scripts/inference.sh

#(spatiallm) [jixu233b@login2.capella SpatialLM]$srun --pty --partition=capella-interactive --ntasks=1 --nodes=1 --time=1:0:0 --cpus-per-task=1 --threads-per-core=1 --mem=20G --gres=gpu:1 bash


#(spatiallm) [jixu233b@login2.capella SpatialLM]$ bash /home/jixu233b/Projects/VLM_3D/SpatialLM/scripts/inference.sh