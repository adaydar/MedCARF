import torch
import math
import sys
import tqdm
import os
from PIL import Image
from torch import nn
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor as meteor
from pycocoevalcap.rouge.rouge import Rouge as rouge
from models import utils
from models.utils import NestedTensor, nested_tensor_from_tensor_list
from models import spc
#from models import spc_k
from models import swavloss as sl
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn.functional as F
from itertools import combinations
import pandas as pd
from utils.correction_report_labels import *
import json
from sentence_transformers import SentenceTransformer, util

check_model = SentenceTransformer('all-mpnet-base-v2')

def contextual_match(gen_list, lab_list):
    scores = []
    for g, l in zip(gen_list, lab_list):
        emb1 = check_model.encode(g, convert_to_tensor=True)
        emb2 = check_model.encode(l, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb1, emb2)
        scores.append(sim.item())
    return scores
    

def merge_caption_tensors(tensor_a, tensor_b):
    merged = []

    for cap_a, cap_b in zip(tensor_a, tensor_b):
        # cap_a and cap_b are Python strings (even inside tensor)
        merged.append(cap_a + " " + cap_b)

    return merged
    
def train_one_epoch(model, cap_model, processor, class_model, criterion, criterionKD, data_loader,
                    optimizer, alpha, beta, device, max_norm, thresholds, tokenizer, config, epoch):
    model.train()
    criterion.train()
    
    class_model.eval()
    epoch_loss = 0.0
    total = len(data_loader)
    cap_model.eval()
    # save_dir_emb = "embeddings_new"
    # os.makedirs(save_dir_emb, exist_ok=True)
    # save_dir_lb = "labels_new"
    # os.makedirs(save_dir_lb, exist_ok=True)
    # num = 0
    # all_embeddings = []
    # all_labels = []
    with tqdm.tqdm(total=total) as pbar:
        for images, masks, com_images, com_masks, caps, cap_masks, image_class, t_img1, t_img2, t_img3, t_img4, t_img5, ip, cap, labels, logit_across_ID in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            com_samples = utils.NestedTensor(com_images, com_masks).to(device)

            caps = caps.to(device)
            image_class = image_class.to(device)
            images = images.to(device)
            labels = labels.to(device)
            t_img1 = t_img1.to(device)
            t_img2 = t_img2.to(device)
            t_img3 = t_img3.to(device)
            t_img4 = t_img4.to(device)
            t_img5 = t_img5.to(device)
            cap_masks = cap_masks.to(device)
            ip = list(ip)
            cap = list(cap)
            labels = list(labels)
            logit_across_ID = logit_across_ID.to(device)
            
            #print(cap)
            
            graph_tensor = spc.process_batch(ip,cap)
            graph_tensor = graph_tensor.to(device)
            #f,_,_,logit = class_model(images)
            
            #print(logit)
            #print(logit_across_ID)
            #for name, p in class_model.named_parameters():
            #   print(name, p.requires_grad)
            
            #thresholded_predictions = 1 * (logit.detach().cpu().numpy() > thresholds)
            
            #print(thresholded_predictions)
            
            thresholded_predictions = 1 * (logit_across_ID.detach().cpu().numpy() > thresholds)
            #phrase_dict = load_phrase_dict("/home/gpuuser5/Akshay/MRG/report_gen/common_file/label_phrases.json")
            #thresholded_predictions = 1 * (logit_across_ID.cpu().numpy() > thresholds)
            #label_report = generate_report_batch(thresholded_predictions, phrase_dict) 
            
            #print(cap)
            #print(label_report)
            
            #print(merged)
            
            outputs = model(com_samples, caps[:, :-1], cap_masks[:, :-1], [thresholded_predictions, tokenizer],graph_tensor)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params:,}")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable_params:,}")
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:]) #+ AuDICoR_loss + loss_ce + kd_loss
            
            loss_value = loss.item()
            epoch_loss += loss_value
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            pbar.update(1)
    return epoch_loss / total

def save_reports_to_file(image_paths, ground_truths, predicted_reports, folder, epoch):
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
    filename = os.path.join(folder, f"knee_reports_epoch{epoch}.txt")

    with open(filename, 'w') as f:
        for idx, (img, gt, pred) in enumerate(zip(image_paths, ground_truths, predicted_reports)):
            f.write(f"Image Path      : {img}\n")
            f.write(f"Ground Truth    : {gt}\n")
            f.write(f"Predicted Report: {pred}\n")
            f.write("\n")

def create_caption_and_mask(start_token, max_length, batch_size):
    caption_template = torch.zeros((batch_size, max_length), dtype=torch.long)
    mask_template = torch.ones((batch_size, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


def compute_scores(gts, res):
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (meteor(), "METEOR"),
        (rouge(), "ROUGE_L")
    ]
    eval_res = {}
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, _ = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


@torch.no_grad()
def evaluate(model, cap_model, processor, class_model, criterion, data_loader, device, config, thresholds, tokenizer, epoch):
    model.eval()
    criterion.eval()
    class_model.eval()
    cap_model.eval()
    total = len(data_loader)
    caption_list = []
    caption_tokens_list = []
    image_paths = []
    reports_dir = "./reports_chest"
    # save_dir_emb = "logits_labels_chest_t"
    # os.makedirs(save_dir_emb, exist_ok=True)
    # all_logits = []
    # all_labels = []

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, _, _, caps, _, image_class, t_1, t_2, t_3, t_4, t_5, ip, _, labels, logit_across_ID in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caption, cap_mask = create_caption_and_mask(
                config.start_token, config.max_position_embeddings, config.batch_size)
            
            ip = list(ip)
            image_paths.extend(ip)
            images = [Image.open(img_path).convert("RGB") for img_path in ip]
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            logit_across_ID = logit_across_ID.to(device)
            with torch.no_grad():  # No gradient needed for inference
                caption_ids = cap_model.generate(**inputs)


            generated_captions = processor.batch_decode(caption_ids, skip_special_tokens=True)
            #save_reports(ip,generated_captions,coarse_dir,epoch)
            #print(generated_captions)
            phrase_dict = load_phrase_dict("common_file/label_phrases.json")
            ground_truth_reports = load_ground_truth_reports("../classification/chest/iu_xray/annotation_labels.json")
            phrase_freq = build_phrase_frequency(ground_truth_reports, phrase_dict)
            
            thresholded_predictions = 1 * (logit_across_ID.cpu().numpy() > thresholds)
            label_report = generate_report_batch(thresholded_predictions, phrase_dict, phrase_freq)           
            merged = merge_caption_tensors(generated_captions, label_report) #New step merging
            
            #print(type(merged))
            #print(type(generated_captions))
            #print(merged)
            scores = contextual_match(generated_captions, label_report)
            
            threshold = 0.4
            selected = []
            for i, score in enumerate(scores):
              if score < threshold:
                selected.append(merged[i])  
                #print(merged[i])            
              else:
                selected.append(generated_captions[i])  
                #print(generated_captions[i])
           
                
            #selected = torch.stack(selected, dim=0)
            
            #print(selected)
            
            graph_tensor = spc.process_batch(ip, selected)
            #graph_tensor = spc.process_batch(ip,merged)
            graph_tensor = graph_tensor.to(device)
            image_class = image_class.to(device)
            #images = images.to(device)

            try:
                for i in range(config.max_position_embeddings - 1):
                    #f,_,_,logit = class_model(image_class)
                    
                    #print(logit)
                    thresholded_predictions = 1 * (logit_across_ID.cpu().numpy() > thresholds)
                    #print
                    #thresholded_predictions = torch.nn.functional.one_hot(torch.argmax(logit, dim=1), num_classes=5)
                    predictions = model(samples.to(device), caption.to(device), cap_mask.to(device),
                                        [thresholded_predictions, tokenizer],graph_tensor)

                    #print(thresholded_predictions)
                    predictions = predictions[:, i, :]
                    predicted_id = torch.argmax(predictions, axis=-1)
                    #print(predicted_id)
                    if i == config.max_position_embeddings - 2:
                        caption_list.extend(caption.cpu().numpy().tolist())
                        caption_tokens_list.extend(caps[:, 1:].cpu().numpy().tolist())
                        break
                    caption[:, i + 1] = predicted_id
                    cap_mask[:, i + 1] = False

            except:
                pass
            pbar.update(1)

        pred = caption_list
        report = caption_tokens_list
        preds_origin = []
        preds = []
        reports = []
        for preds_sentence in pred:
            single_sentence = list()
            for item in preds_sentence:
                single_sentence.append(item)
                if item==2:
                    preds_origin.append(single_sentence)
                    continue

        # all_logits_tensor = torch.cat(all_logits, dim=0)  # concatenate along the batch dimension
        # np.save(os.path.join(save_dir_emb, f"logits{epoch}.npy"), all_logits_tensor.cpu().numpy())
        # all_labels_tensor = torch.cat(all_labels, dim=0)
        # np.save(os.path.join(save_dir_emb, f"labels{epoch}.npy"), all_labels_tensor.cpu().numpy())
        for preds_sentence in pred:
            preds.append([item for item in preds_sentence if item not in [config.start_token, config.end_token, 0]])
        
        for reports_sentence in report:
            reports.append([item for item in reports_sentence if item not in [config.start_token, config.end_token, 0]])
        
        ground_truth = [tokenizer.decode(item) for item in reports]
        pred_result = [tokenizer.decode(item) for item in preds]
        # print(ground_truth)
        # print(pred_result)
        save_reports_to_file(image_paths, ground_truth, pred_result,reports_dir, epoch)
        
        val_met = compute_scores({i: [gt] for i, gt in enumerate(ground_truth)},
                                 {i: [re] for i, re in enumerate(pred_result)})
        return val_met

