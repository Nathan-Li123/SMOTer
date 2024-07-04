CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --num-gpus 4 --config-file configs/BYTE_BENSMOT_FPN.yaml
CUDA_VISIBLE_DEVICES=0 python train_net.py --num-gpus 1 --config-file configs/BYTE_BENSMOT_FPN.yaml --eval-only path/to/weight
python eval_vu.py