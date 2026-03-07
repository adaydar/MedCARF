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
#from models import spc
from models import spc_k
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

def unlikelihood_loss(logits, target_ids, pad_token_id=None):
    """
    logits: [batch_size, seq_len, vocab_size] - raw outputs from the model
    target_ids: [batch_size, seq_len] - ground truth token IDs
    """
    batch_size, seq_len, vocab_size = logits.size()

    # Get probabilities
    probs = F.softmax(logits, dim=-1)

    # Build unlikelihood mask: mark positions where token was already generated earlier in sequence
    unlikelihood_mask = torch.zeros_like(probs)  # [B, T, V]

    for b in range(batch_size):
        seen = set()
        for t in range(seq_len):
            token_id = target_ids[b, t].item()
            if token_id in seen and (pad_token_id is None or token_id != pad_token_id):
                # Penalize repeated tokens
                unlikelihood_mask[b, t, token_id] = 1.0
            seen.add(token_id)

    # Select probabilities of repeated tokens
    p_repeats = torch.sum(probs * unlikelihood_mask, dim=-1)  # [B, T]

    # Apply unlikelihood loss: -log(1 - p)
    # Clamp to avoid log(0)
    epsilon = 1e-8
    ul_loss = -torch.log(1.0 - p_repeats + epsilon)

    # Only consider timesteps where we have unlikelihood signal
    mask = (unlikelihood_mask.sum(dim=-1) > 0).float()  # [B, T]
    ul_loss = ul_loss * mask

    return ul_loss.sum() / (mask.sum() + epsilon)
    
def train_one_epoch(model, cap_model, processor, class_model, criterion, criterionKD, data_loader,
                    optimizer, alpha, beta, device, max_norm, tokenizer, config, epoch):
    model.train()
    criterion.train()    
    class_model.eval()
    epoch_loss = 0.0
    total = len(data_loader)
    cap_model.eval()
    save_dir_emb = "embeddings_new"
    os.makedirs(save_dir_emb, exist_ok=True)
    save_dir_lb = "labels_new"
    os.makedirs(save_dir_lb, exist_ok=True)
    loss_function = torch.nn.CrossEntropyLoss(weight=None)
    num = 0
    all_embeddings = []
    all_labels = []
    with tqdm.tqdm(total=total) as pbar:
        for images, masks, com_images, com_masks, caps, cap_masks, image_class,ip, cap, labels,tab_f in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            com_samples = utils.NestedTensor(com_images, com_masks).to(device)

            caps = caps.to(device)
            image_class = image_class.to(device)
            labels = labels.to(device)
            cap_masks = cap_masks.to(device)
            ip = list(ip)
            cap = list(cap)
            graph_tensor = spc_k.process_batch(ip,cap)
            graph_tensor = graph_tensor.to(device)
            logit, f = class_model(image_class)
            thresholded_predictions = torch.nn.functional.one_hot(torch.argmax(logit, dim=1), num_classes=5)
            outputs = model(com_samples, caps[:, :-1], cap_masks[:, :-1], [thresholded_predictions, tokenizer],tab_f,graph_tensor)
            num += 1 
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:]) + unlikelihood_loss(outputs, caps[:, 1:])
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

    #all_embeddings = np.vstack(all_embeddings)
    #save_path = os.path.join(save_dir_emb, f"epoch_{epoch}.npy")
    #df = pd.DataFrame({'label': all_labels})
    #csv_path = os.path.join(save_dir_lb, f"epoch_{epoch}.csv")
    #if epoch%5 == 0:
    #    np.save(save_path, all_embeddings)
    #    df.to_csv(csv_path, index=False)
    return epoch_loss / total

def save_reports_to_file(image_paths, ground_truths, predicted_reports, folder, epoch):
    """
    Saves image paths, ground truth reports, and predicted reports to a text file inside a specified folder.

    Args:
        image_paths (List[str]): List of image file paths.
        ground_truths (List[str]): List of ground truth report strings.
        predicted_reports (List[str]): List of predicted report strings.
        epoch (int): Current epoch number.
        folder (str): Directory name to store the reports.
    """
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
    filename = os.path.join(folder, f"knee_reports{epoch}.txt")

    with open(filename, 'w') as f:
        for idx, (img, gt, pred) in enumerate(zip(image_paths, ground_truths, predicted_reports)):
            f.write(f"Image Path      : {img}\n")
            f.write(f"Ground Truth    : {gt}\n")
            f.write(f"Predicted Report: {pred}\n")
    
    #print(f"Reports saved to: {filename}")

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
def evaluate(model, cap_model, processor, class_model, criterion, data_loader, device, config, tokenizer, epoch):
    model.eval()
    criterion.eval()
    class_model.eval()
    cap_model.eval()
    total = len(data_loader)
    caption_list = []
    caption_tokens_list = []
    image_paths = []
    reports_dir = "./reports_knee"
    save_dir_emb = "embeddings_knee_t"
    os.makedirs(save_dir_emb, exist_ok=True)
    save_dir_lb = "labels_knee_t"
    os.makedirs(save_dir_lb, exist_ok=True)
    all_embeddings = []
    all_labels = []

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, _, _, caps, _, image_class, ip, _, _,tab_f in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caption, cap_mask = create_caption_and_mask(
                config.start_token, config.max_position_embeddings, config.batch_size)
            
            ip = list(ip)
            image_paths.extend(ip)
            images = [Image.open(img_path).convert("RGB") for img_path in ip]
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            with torch.no_grad():  # No gradient needed for inference
                caption_ids = cap_model.generate(**inputs)


            generated_captions = processor.batch_decode(caption_ids, skip_special_tokens=True)
            #save_reports(ip,generated_captions,coarse_dir,epoch)
            graph_tensor = spc_k.process_batch(ip,generated_captions)
            graph_tensor = graph_tensor.to(device)
            image_class = image_class.to(device)
            logit, f = class_model(image_class)
            label = torch.nn.functional.one_hot(torch.argmax(logit, dim=1), num_classes=5)
            f = f.cpu().numpy()
            label = label.cpu().numpy()
            all_embeddings.extend(f)
            all_labels.extend(label)

            try:
                for i in range(config.max_position_embeddings - 1):
                    logit, f = class_model(image_class)
                    thresholded_predictions = torch.nn.functional.one_hot(torch.argmax(logit, dim=1), num_classes=5)
                    predictions = model(samples.to(device), caption.to(device), cap_mask.to(device),
                                        [thresholded_predictions, tokenizer],tab_f,graph_tensor)

                    predictions = predictions[:, i, :]
                    predicted_id = torch.argmax(predictions, axis=-1)
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

        all_embeddings = np.vstack(all_embeddings)
        save_path = os.path.join(save_dir_emb, f"embd{epoch}.npy")
        df = pd.DataFrame({'label': all_labels})
        csv_path = os.path.join(save_dir_lb, f"labels{epoch}.csv")
        np.save(save_path, all_embeddings)
        df.to_csv(csv_path, index=False)
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

