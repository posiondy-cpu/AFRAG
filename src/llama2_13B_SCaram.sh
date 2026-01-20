#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "当前可见 GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi
# 启动 Elasticsearch 并记录 PID
export DATA_PATH=/home/bingxing2/home/scx8jw3/aram/elasticsearch-7.9.2/data/
mkdir -p $DATA_PATH
nohup ../elasticsearch-7.9.2/bin/elasticsearch -Epath.data=$DATA_PATH > es.log 2>&1 &
ES_PID=$!
echo "Elasticsearch PID: $ES_PID"

# 等待服务启动
sleep 15

# 加载环境
module load miniforge3/24.1 compilers/cuda/12.1 cudnn/8.9.5.29_cuda12.x compilers/gcc/11.3.0 nccl/2.19.1-1_cuda12.1
source /home/bingxing2/apps/miniforge3/24.1.2/etc/profile.d/conda.sh
conda activate aram

# 执行任务
# python main.py -c ../config/Llama2-13b-chat/2WikiMultihopQA/ARAM_SC.json
python main.py -c ../config/Llama2-13b-chat/HotpotQA/ARAM_SC.json
# python main.py -c ../config/Llama2-13b-chat/IIRC/ARAM_SC.json
python main.py -c ../config/Llama2-13b-chat/StrategyQA/ARAM_SC.json

# ✅ 自动关闭 Elasticsearch，释放节点资源
echo "任务完成，杀掉 Elasticsearch..."
kill -9 $ES_PID
