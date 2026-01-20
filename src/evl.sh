#!/bin/sh


export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "当前可见 GPU: $CUDA_VISIBLE_DEVICES"

nvidia-smi

module load miniforge3/24.1 compilers/cuda/12.1 cudnn/8.9.5.29_cuda12.x compilers/gcc/11.3.0 nccl/2.19.1-1_cuda12.1
source /home/bingxing2/apps/miniforge3/24.1.2/etc/profile.d/conda.sh
conda activate aram

# ROOT_DIR="aram"
# for dir in "$ROOT_DIR"/*/*/; do
#     if [ -d "$dir" ]; then
#         echo "处理目录：$dir"
#         python evaluate.py --dir "$dir" 
#     fi
# done

# ROOT_DIR="aramSC"
# for dir in "$ROOT_DIR"/*/*/; do
#     if [ -d "$dir" ]; then
#         echo "处理目录：$dir"
#         python evaluate.py --dir "$dir" 
#     fi
# done



ROOT_DIR="FLARE"
for dir in "$ROOT_DIR"/*; do
    if [ -d "$dir" ]; then
        echo "处理目录：$dir"
        python evaluate.py --dir "$dir" 
    fi
done