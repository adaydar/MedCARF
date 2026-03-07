#!/bin/bash
#export CUDA_VISIBLE_DEVICES=3
python3 main.py --batch_size 8 --image_size 300 --vocab_size 760 --theta 0.4 --gamma 0.4 --beta 1.0 --delta 0.01 --dataset_name iu_xray --anno_path ../classification/chest/iu_xray/annotation_labels.json --data_dir ../classification/chest/iu_xray/images --mode test --knowledge_prompt_path ./common_files/knowledge_prompt_mimic.pkl --test_path best_epoch.pth

