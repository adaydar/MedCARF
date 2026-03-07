#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=2
python3 main.py --epochs 50 --lr_backbone 1e-5 --lr 1e-4 --batch_size 2 --image_size 300 --vocab_size 760 --theta 0.4 --gamma 0.4 --beta 1.0 --delta 0.01 --dataset_name knee_xray --t_model_weight_path ./weight_path/iutmodel_new.pth --anno_path ../kneeXray/annotation_255_tab1.json --data_dir ../kneeXray/OAI_dataset_255/images --mode train

