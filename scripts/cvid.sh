CUDA_VISIBLE_DEVICES=1,2,3 python train_net.py --num-gpus 3 --config-file configs/BYTE_CVID_FPN.yaml
CUDA_VISIBLE_DEVICES=0 python train_net.py --num-gpus 1 --config-file configs/BYTE_CVID_FPN.yaml --eval-only MODEL.WEIGHTS output/GTR_MOT/BYTE_CVID_FPN/model_0024999.pth
python eval_vu.py