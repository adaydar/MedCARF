# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from networks import define_network
from torch.optim import lr_scheduler
import functools
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import torch.nn.functional as F
from itertools import combinations
import pandas as pd

class Net(nn.Module):
    def __init__(self, lr, weight_decay, init_type, gpu_ids, network,
                 pretrain, avg, weight, milestones, truncated, alpha, num_classes):
        super(Net, self).__init__()
        self.lr = lr
        self.avg = avg
        self.alpha = alpha
        self.weight = weight
        self.gpu_ids = gpu_ids
        self.network = network
        self.truncated = truncated
        self.milestones = milestones

        self.model = define_network(init_type, gpu_ids, network, pretrain, avg, weight, truncated, num_classes)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=weight_decay, momentum=0.9)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones, gamma=0.1, last_epoch=-1)
        self.alpha = torch.nn.Parameter(torch.tensor(0.07))
        self.loss1 = 0
        self.loss2 = 0
        self.loss = 0
        self.probs = None
        self.feat = None

    def forward(self, mode='train'):
        if self.avg == 0:
            # PCAN
            if self.weight == 1:
                self.f, self.predicted_all, self.weight_all, self.logit = self.model(self.im)
                self.f1, self.predicted_all1, self.weight_all1, self.logit1 = self.model(self.im1)
                self.f2, self.predicted_all2, self.weight_all2, self.logit2 = self.model(self.im2)
                self.f3, self.predicted_all3, self.weight_all3, self.logit3 = self.model(self.im3)
                self.f4, self.predicted_all4, self.weight_all4, self.logit4 = self.model(self.im4)
                self.f5, self.predicted_all5, self.weight_all5, self.logit5 = self.model(self.im5)
                ori_features = list(torch.unbind(self.f, dim=0))
                tr1_f = list(torch.unbind(self.f1, dim=0))
                tr2_f = list(torch.unbind(self.f2, dim=0))
                tr3_f = list(torch.unbind(self.f3, dim=0))
                tr4_f = list(torch.unbind(self.f4, dim=0))
                tr5_f = list(torch.unbind(self.f5, dim=0))
                transformed_probs = [self.logit1, self.logit2, self.logit3, self.logit4, self.logit5]
                transformed_features = [tr1_f, tr2_f, tr3_f, tr4_f, tr5_f]
                AuDICoR_loss = audicor_loss(self.logit, transformed_probs, ori_features, transformed_features, self.alpha)
                #print("audicor",AuDICoR_loss)
                self.loss = AuDICoR_loss
                self.probs = self.logit
                self.feat = self.f
                #print("total",self.loss)
            
            # Multiple instance learning
            else:
                self.predicted_all, self.final_predicted = self.model(self.im)                
                self.weight_all = self.predicted_all
                self.loss = self.criterion(self.final_predicted.float(), self.label.float())

        # Global average pooling
        else:
            self.final_predicted = self.model(self.im)
            self.loss = self.criterion(self.final_predicted.float(), self.label.float())

    def set_input(self, x,a,b,c,d,e):
        self.im = x
        self.im1 = a
        self.im2 = b
        self.im3 = c
        self.im4 = d
        self.im5 = e
        self.forward()
        #self.w = self.im.size()[-1] // 32 if self.truncated == 1 else self.im.size()[-1] // 16

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def predicted_val(self):
        a = self.probs
        b = self.feat
        return a, b

    def predicted_test(self):
        a = self.logit
        b = self.predicted_all.cpu().numpy()
        c = self.label.cpu().numpy()
        d = self.weight_all.cpu().numpy()
        e = self.im.cpu().numpy()

        return a, b, c, d, e

    def update_learning_rate(self):
        self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']

    def print_networks(self):
        num_params = 0
        for param in self.model.parameters():
            num_params += param.numel()
        print('Total number of parameters : %.3f M' % (num_params / 1e6))

def custom_contrastive_loss(transformed_features, near_cluster_dict, far_cluster_dict):
    total_loss = 0.0
    batch_size = len(near_cluster_dict)

    for anchor_idx in range(batch_size):
        near = near_cluster_dict.get(anchor_idx, [])
        far = far_cluster_dict.get(anchor_idx, [])

        near_feats = [transformed_features[t_idx][img_idx] for img_idx, t_idx in near]
        far_feats = [transformed_features[t_idx][img_idx] for img_idx, t_idx in far]

        pos_term = 0.0
        neg_term = 0.0

        # Positive pairs: from near cluster
        if len(near_feats) > 1:
            for feat1, feat2 in combinations(near_feats, 2):
                dist = F.pairwise_distance(feat1.unsqueeze(0), feat2.unsqueeze(0))  # shape [1]
                sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0))  # shape [1]
                sim = (sim + 1) / 2  # normalize to [0, 1]
                loss = dist * sim  # pull closer, weighted by current similarity
                pos_term += loss.item()

        # Negative pairs: near vs. far cluster
        if len(near_feats) > 0 and len(far_feats) > 0:
            for near_feat in near_feats:
                for far_feat in far_feats:
                    dist = F.pairwise_distance(near_feat.unsqueeze(0), far_feat.unsqueeze(0))  # shape [1]
                    sim = F.cosine_similarity(near_feat.unsqueeze(0), far_feat.unsqueeze(0))  # shape [1]
                    sim = (sim + 1) / 2  # normalize to [0, 1]
                    loss = dist * sim  # push apart, weighted by current similarity
                    neg_term += loss.item()

        total_loss += pos_term - neg_term

    return total_loss 


def audicor_loss(image_probs: torch.Tensor, transformed_probs: list, original_features: list, transformed_features: list, alpha: float = 0.07):
    batch_size, num_classes = image_probs.shape

    # Compute cross-entropy loss between each image and its transformations
    loss_values = []
    image_indices = []

    for i, transform in enumerate(transformed_probs):
        loss = F.cross_entropy(transform, image_probs, reduction='none')  # Compute loss per image
        loss_values.append(loss)
        image_indices.extend([(img_idx, i) for img_idx in range(batch_size)])  # Track original image index

    loss_values = torch.stack(loss_values, dim=1)  # Shape: [8, 5]
    loss_values = loss_values.view(-1).detach().cpu().numpy()  # Flatten to [40]
    image_indices = np.array(image_indices)  # Convert to NumPy array

    # KMeans clustering (2 clusters)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(loss_values.reshape(-1, 1))
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_.flatten()

    # Identify near and far clusters
    if cluster_centers[0] < cluster_centers[1]:
        near_cluster, far_cluster = 0, 1
    else:
        near_cluster, far_cluster = 1, 0

    # Compute cluster-wise loss sums
    near_loss = np.sum(loss_values[labels == near_cluster])
    far_loss = np.sum(loss_values[labels == far_cluster])

    # Find the point closest to the near cluster center
    near_cluster_indices = np.where(labels == near_cluster)[0]
    closest_index = near_cluster_indices[np.argmin(np.abs(loss_values[near_cluster_indices] - cluster_centers[near_cluster]))]
    closest_image_index, _ = image_indices[closest_index]
    # print(f"Closest point to near cluster center is from Image {closest_image_index}")

    # Print image indices and transformation indices for each cluster
    near_cluster_images = image_indices[labels == near_cluster]
    far_cluster_images = image_indices[labels == far_cluster]
    # print(f"Near Cluster (Cluster {near_cluster}): {near_cluster_images}")
    # print(f"Far Cluster (Cluster {far_cluster}): {far_cluster_images}")
    near_cluster_dict = {i: [] for i in range(batch_size)}
    far_cluster_dict = {i: [] for i in range(batch_size)}

    for img_idx, t_idx in near_cluster_images:
        near_cluster_dict[img_idx].append((img_idx, t_idx))
    for img_idx, t_idx in far_cluster_images:
        far_cluster_dict[img_idx].append((img_idx, t_idx))

    # Contrastive loss computation
    anchor_features = original_features[closest_image_index]
    positive_features = torch.stack([transformed_features[t_idx][img_idx] for img_idx, t_idx in near_cluster_images])
    negative_features = torch.stack([transformed_features[t_idx][img_idx] for img_idx, t_idx in far_cluster_images])

    contrastive_loss_value = custom_contrastive_loss(transformed_features, near_cluster_dict, far_cluster_dict)
    #print(f"Contrastive Loss: {contrastive_loss_value}")

    # Loss function
    final_loss = ((alpha * near_loss + (1 - alpha) * far_loss) +  contrastive_loss_value) / 40
    #print(f"Final Loss: {final_loss}")

    return final_loss