CUDA_VISIBLE_DEVICES=1,2,3,5

python main.py -c ../config/Llama2-13b-chat/2WikiMultihopQA/DRAGIN.json
python main.py -c ../config/Llama2-13b-chat/HotpotQA/DRAGIN.json
python main.py -c ../config/Llama2-13b-chat/IIRC/DRAGIN.json
python main.py -c ../config/Llama2-13b-chat/StrategyQA/DRAGIN.json
python main.py -c ../config/Llama2-13b-chat/HotpotQA/ARAM.json
python main.py -c ../config/Llama2-13b-chat/IIRC/ARAM.json
python main.py -c ../config/Llama2-13b-chat/HotpotQA/ARAM_SC.json
python main.py -c ../config/Llama2-13b-chat/IIRC/ARAM_SC.json