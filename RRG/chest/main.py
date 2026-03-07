import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import argparse
import os
from torch import nn
from torch.optim import AdamW, SGD
from models import utils, caption, bmloss
from datasets import xray
from transformers import BlipProcessor, BlipForConditionalGeneration
#from adapter_transformers import AdapterConfig
from peft import get_peft_model, LoraConfig, TaskType
from utils.engine import train_one_epoch, evaluate
from models.model import swin_tiny_patch4_window7_224 as create_model
from networks import define_network as create_model
from model import Net
from utils.stloss import SoftTarget
from setproctitle import setproctitle
from torchvision import datasets,transforms,models


def build_diagnosisbot(device, num_classes, detector_weight_path):
    init_type = "normal"
    network = -1
    gpu_ids = 0
    model = torch.load(detector_weight_path, weights_only = False)
    #model = torch.load('model_auc.pkl')
    model = create_model(init_type,gpu_ids,network).to(device)
    for p in model.parameters():
        p.requires_grad = False 
    #model.load_state_dict(state_dict)
    
    
    return model

def main(config):
    print(config)
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    if os.path.exists(config.thresholds_path):
        with open(config.thresholds_path, "rb") as f:
            thresholds = pickle.load(f)

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cap_model = BlipForConditionalGeneration.from_pretrained("umarigan/blip-image-captioning-base-chestxray-finetuned")
    processor = BlipProcessor.from_pretrained("umarigan/blip-image-captioning-base-chestxray-finetuned")
    cap_model.to(device)
    detector = build_diagnosisbot(device, config.num_classes, config.detector_weight_path)
    detector.to(device)
    model, criterion = caption.build_model(config)
    criterionKD = SoftTarget(4.0)
    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    trainable_parameters = sum(p.numel() for p in cap_model.parameters() if p.requires_grad)
    print(f"number of parameters of blip model: {trainable_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    
    alpha = torch.nn.Parameter(torch.tensor(0.07))
    beta = torch.nn.Parameter(torch.tensor(0.5))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    dataset_train = xray.build_dataset(config, processor, mode='training', anno_path=config.anno_path, data_dir=config.data_dir,
                                       dataset_name=config.dataset_name, image_size=config.image_size,
                                       theta=config.theta, gamma=config.gamma, beta=config.beta)
    dataset_val = xray.build_dataset(config, processor, mode='validation', anno_path=config.anno_path, data_dir=config.data_dir,
                                     dataset_name=config.dataset_name, image_size=config.image_size,
                                     theta=config.theta, gamma=config.gamma, beta=config.beta)
    dataset_test = xray.build_dataset(config, processor, mode='test', anno_path=config.anno_path, data_dir=config.data_dir,
                                      dataset_name=config.dataset_name, image_size=config.image_size,
                                      theta=config.theta, gamma=config.gamma, beta=config.beta)
                                      
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")
    print(f"Test: {len(dataset_test)}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers,
        collate_fn=dataset_train.collate_fn)

    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False,
                                 collate_fn=dataset_val.collate_fn)

    data_loader_test = DataLoader(dataset_test, config.batch_size,
                                  sampler=sampler_test, drop_last=False,
                                  collate_fn=dataset_test.collate_fn)
    
    # if os.path.exists(config.checkpoint):
    #     print("inside checkpoint")
    #     print(config.checkpoint)
    #     weights_dict = torch.load(config.checkpoint, map_location='cpu')['model']
    #     opt = torch.load(config.checkpoint, map_location='cpu')['optimizer']
    #     lr = torch.load(config.checkpoint, map_location='cpu')['lr_scheduler']
    #     model.load_state_dict(weights_dict)
    #     optimizer.load_state_dict(opt)
    #     lr_scheduler.load_state_dict(lr)

    filename = "result_chest.txt"
    if True:
        print("Start Training..")
        #tmodel = build_tmodel(config, device)
        for epoch in range(config.start_epoch, config.epochs):
            print(f"Epoch: {epoch}")
            epoch_loss = train_one_epoch(
                model, cap_model, processor, detector, criterion, criterionKD, data_loader_train, optimizer, alpha, beta, device,
                config.clip_max_norm, thresholds=thresholds, tokenizer=dataset_train.tokenizer, config=config, epoch=epoch)
            lr_scheduler.step()
            print(f"Training Loss: {epoch_loss}")

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, config.dataset_name + "chest_weight_epoch" + str(epoch) + "_.pth")

            validate_result = evaluate(model, cap_model, processor, detector, criterion, data_loader_val, device, config,
                                       thresholds=thresholds, tokenizer=dataset_val.tokenizer,epoch=epoch)
            print(f"validate_result: {validate_result}")
            test_result = evaluate(model, cap_model, processor, detector, criterion, data_loader_test, device, config,
                                   thresholds=thresholds, tokenizer=dataset_test.tokenizer,epoch=epoch)
            print(f"test_result: {test_result}")

            with open(filename, "a") as file:
                file.write(f"con_epoch {epoch}\n")
                file.write(f"con_validate result: {validate_result}\n")
                file.write(f"con_test result: {test_result}\n\n")


    if False:
        if os.path.exists(config.test_path):
            weights_dict = torch.load(config.test_path, map_location='cpu')['model']
            model.load_state_dict(weights_dict, strict=False)

        print("Start Testing..")
        test_result = evaluate(model, cap_model, processor, detector, criterion, data_loader_test, device, config,
                                   thresholds=thresholds, tokenizer=dataset_test.tokenizer)
        print(f"test_result: {test_result}")


if __name__ == "__main__":
    setproctitle("Keshav")
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_drop', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--pre', type=bool, default=True)
    parser.add_argument('--lr_decay_epoch', type=int, default=5)

    # Backbone
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--position_embedding', type=str, default='sine')
    parser.add_argument('--dilation', type=bool, default=True)
    # Basic
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--clip_max_norm', type=float, default=0.1)

    # Transformer
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--max_position_embeddings', type=int, default=128)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--vocab_size', type=int, default=4253)
    parser.add_argument('--start_token', type=int, default=1)
    parser.add_argument('--end_token', type=int, default=2)

    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--pre_norm', type=int, default=True)

    # diagnosisbot
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--thresholds_path', type=str, default="./datasets/thresholds.pkl")
    parser.add_argument('--detector_weight_path', type=str, default="../classification/chest/predictor_module/runs/1_0.01_16_-1_-1_0_1.0_337843/model_final.pkl")
    parser.add_argument('--knowledge_prompt_path', type=str, default="common_file/knowledge_prompt_mimic.pkl")

    # ADA
    parser.add_argument('--theta', type=float, default=0.4)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=1.0)

    # Delta
    parser.add_argument('--delta', type=float, default=0.01)

    # Dataset
    parser.add_argument('--image_size', type=int, default=300)
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr')
    parser.add_argument('--anno_path', type=str, default='../dataset/mimic_cxr/annotation.json')
    parser.add_argument('--data_dir', type=str, default='../dataset/mimic_cxr/images300')
    parser.add_argument('--limit', type=int, default=-1)

    # mode
    parser.add_argument('--mode', type=str, default="train")
    # parser.add_argument('--test_path', type=str, default="iu_xray2_weight_epoch7_.pth")
    # parser.add_argument('--checkpoint', type=str, default="iu_xraycon_chest_weight_epoch10_.pth")

    config = parser.parse_args()
    main(config)
