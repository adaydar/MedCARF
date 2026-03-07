#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=2
python3 main.py --epochs 30 --lr_backbone 1e-5 --lr 1e-4 --batch_size 8 --image_size 300 --vocab_size 760 --theta 0.4 --gamma 0.4 --beta 1.0 --delta 0.01 --dataset_name iu_xray --anno_path ../classification/chest/iu_xray/annotation_labels.json --data_dir ../classification/chest/iu_xray/images --mode train

