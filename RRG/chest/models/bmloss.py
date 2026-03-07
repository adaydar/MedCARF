import torch
from torch import nn
import torch.nn.functional as F
from .utils import NestedTensor, nested_tensor_from_tensor_list
import cv2
#from transformers import ViTModel, ViTImageProcessor
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow
from PIL import Image
import math
import sys
import tqdm
from scipy.optimize import linear_sum_assignment
import numpy as np


class BML(nn.Module):
    def __init__(self):
        super().__init__()

    def detect_heated_regions(self,heatmap):
        if len(heatmap.shape) == 3:
            gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        else:
            gray = heatmap

        # Apply threshold to identify heated regions
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cv2_imshow(binary)
        # Find contours
        contours, _ = cv2.findContours(binary.astype(np.uint8),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        #cv2_imshow(cv2.drawContours(heatmap, contours, -1, (0, 255, 0), 2))
        # Get bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

        return boxes
    
    def iou(self,box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def compute_bipartite_matching_loss(self,gt_boxes,pred_boxes):
        cost_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt in enumerate(gt_boxes):
            for j, pred in enumerate(pred_boxes):
                cost_matrix[i, j] = 1 - self.iou(gt, pred)  # Use 1 - IoU as cost

        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_loss = cost_matrix[row_ind, col_ind].sum()

        return total_loss
    
    def bml(self,img1,img2):
        h1 = cv2.imread(img1)
        h2 = cv2.imread(img2)
        boxes1 = self.detect_heated_regions(h1)
        boxes2 = self.detect_heated_regions(h2)
        return self.compute_bipartite_matching_loss(boxes1,boxes2)

    def loss_calc(self,indices,images):
        loss = 0
        for i in range(8):
            ori_img = images[i].replace("images","resnet/images_seg").replace(".png","_h.png")
            best_t_img = images[i].replace("images","resnet/images_seg").replace(".png",f"_t{indices[i]}_h.png")
            loss += self.bml(ori_img,best_t_img)
        
        return loss/8
    
    def forward(self,original_probs, original_labels, 
                                 tr1_b_probs, tr1_b_labels, 
                                 tr2_b_probs, tr2_b_labels, 
                                 tr3_b_probs, tr3_b_labels, 
                                 tr4_b_probs, tr4_b_labels, 
                                 tr5_b_probs, tr5_b_labels,images):

        """
        Find the best transformed image index for each image in the batch.

        :param original_probs: (numpy array) Class probabilities of the original images (shape: [batch_size, 14])
        :param original_labels: (numpy array) One-hot encoded labels of the original images (shape: [batch_size, 14])
        :param tr1_b_probs, tr1_b_labels: Probabilities & labels for transformed batch 1 (shape: [batch_size, 14])
        :param tr2_b_probs, tr2_b_labels: Probabilities & labels for transformed batch 2 (shape: [batch_size, 14])
        :param tr3_b_probs, tr3_b_labels: Probabilities & labels for transformed batch 3 (shape: [batch_size, 14])
        :param tr4_b_probs, tr4_b_labels: Probabilities & labels for transformed batch 4 (shape: [batch_size, 14])
        :param tr5_b_probs, tr5_b_labels: Probabilities & labels for transformed batch 5 (shape: [batch_size, 14])

        :return: List of best transformed image indices (size: batch_size)
        """
        #print(len(images))
        batch_size = len(images)
        best_indices = []  # Store best transformed image index for each original image

        # Group transformed batches for easy iteration
        transformed_probs_list = [tr1_b_probs, tr2_b_probs, tr3_b_probs, tr4_b_probs, tr5_b_probs]
        transformed_labels_list = [tr1_b_labels, tr2_b_labels, tr3_b_labels, tr4_b_labels, tr5_b_labels]

        for i in range(batch_size):
            orig_prob = original_probs[i]  # Shape: (14,)
            orig_label = original_labels[i]  # Shape: (14,)

            transformed_probs = [trans[i] for trans in transformed_probs_list]  # Get transformed images for image `i`
            transformed_labels = [trans[i] for trans in transformed_labels_list]

            # Step 1: Identify the important class in the original image
            valid_indices = np.where(orig_label == 1)[0]  # Get indices where label is 1
            if len(valid_indices) == 0:
                best_indices.append(-1)  # No valid class found, return -1 as error case
                continue

            important_label = valid_indices[np.argmax(orig_prob[valid_indices].detach().cpu().numpy())]

            # Step 2: Find transformed images with the same important label
            candidate_indices = []
            candidate_scores = []

            for t_idx, (trans_prob, trans_label) in enumerate(zip(transformed_probs, transformed_labels)):
                valid_trans_indices = np.where(trans_label == 1)[0]
                
                if important_label in valid_trans_indices:
                    candidate_indices.append(t_idx)
                    candidate_scores.append(trans_prob[important_label])

            # If candidates are found, return the one with the highest probability
            if candidate_indices:
                candidate_scores = torch.tensor(candidate_scores)  # Convert list to tensor
                best_indices.append(candidate_indices[torch.argmax(candidate_scores).item()])
                continue

            # Step 3: If no match in Step 2, find the transformed image with the lowest probability for the important label
            min_prob_index = np.argmin([trans_prob[important_label].detach().cpu().numpy() for trans_prob in transformed_probs])
            best_indices.append(min_prob_index)


        loss = self.loss_calc(best_indices,images)

        return loss

    




