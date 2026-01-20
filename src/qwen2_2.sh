#!/bin/sh

# nohup ./elasticsearch-7.9.1/bin/elasticsearch &


# python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index

# nohup ./elasticsearch-7.9.1/bin/elasticsearch  & > es.log 2>&1 &

# until curl -s http://localhost:9207 > /dev/null; do sleep 2; done
# sleep 60
set -e
# export LD_LIBRARY_PATH=/home/bingxing2/home/scx8jw3/.conda/envs/aram2/lib/python3.10/site-packages/torch/lib/libtorch_global_deps.so
# 第一步构建wiki index
# python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki 
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "当前可见 GPU: $CUDA_VISIBLE_DEVICES"

nvidia-smi

nohup ../elasticsearch-7.9.1/bin/elasticsearch  & > es.log 2>&1 &
sleep 30
source /home/bingxing2/apps/miniforge3/24.1.2/etc/profile.d/conda.sh
conda activate aram2
export PYTHONNOUSERSITE=1

python main.py -c ../config/Llama2-13b-chat/2WikiMultihopQA/ARAM.json
# python main.py -c ../config/qwen2-7b/2WikiMultihopQA/FL-RAG.json
# python main.py -c ../config/qwen2-7b/IIRC/SR-RAG.json
# python main.py -c ../config/qwen2-7b/StrategyQA/SR-RAG.json

# 67 ——>1 ---> FS
# 68 __>2 ————>wo
# 69 ——> 3 -->SR
# 13 __>3 __>FLARE