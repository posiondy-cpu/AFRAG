#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "当前可见 GPU: $CUDA_VISIBLE_DEVICES"
# cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
nvidia-smi
cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
export DATA_PATH=/home/bingxing2/home/scx8jw3/aram/elasticsearch-7.9.2/data/
mkdir -p $DATA_PATH
nohup ../elasticsearch-7.9.2/bin/elasticsearch -Epath.data=$DATA_PATH   & > es.log 2>&1 &
sleep 10

if [[ "$cuda_version" == "12.2" ]]; then
    module load miniforge3/24.1 compilers/cuda/12.1   compilers/gcc/11.3.0 
    source activate aram1
    python main.py -c ../config/qwen2-7b/2WikiMultihopQA/ARAM.json
    # python main.py -c ../config/qwen2-7b/HotpotQA/ARAM.json
    # python main.py -c ../config/qwen2-7b/IIRC/ARAM.json
    # python main.py -c ../config/qwen2-7b/StrategyQA/ARAM.json

elif [[ "$cuda_version" == "11.6" ]]; then
    module load miniforge3/24.1 compilers/cuda/12.4   compilers/gcc/11.3.0 
    source activate aram1
    python main.py -c ../config/qwen2-7b/2WikiMultihopQA/ARAM.json
    # python main.py -c ../config/qwen2-7b/HotpotQA/ARAM.json
    # python main.py -c ../config/qwen2-7b/IIRC/ARAM.json
    # python main.py -c ../config/qwen2-7b/StrategyQA/ARAM.json
else
    echo "❌ Unsupported CUDA version: $cuda_version"
    exit 1
fi


# python main.py -c ../config/qwen2-7b/2WikiMultihopQA/ARAM.json
# python main.py -c ../config/qwen2-7b/HotpotQA/ARAM_SC.json
# python main.py -c ../config/qwen2-7b/IIRC/ARAM_SC.json
# python main.py -c ../config/qwen2-7b/StrategyQA/ARAM_SC.json
