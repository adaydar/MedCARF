import os
import shutil
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import classification_report
import torch
#!pip install monai
from monai.transforms import Activations, AsDiscrete
from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import *
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from tqdm import tqdm
import pandas as pd
from torchsummary import summary
from torchvision import datasets,transforms,models
import sys
from torchvision.transforms import functional as F
import torch.nn.functional as F1
from sklearn.cluster import KMeans
from itertools import combinations

# Fix all random seeds.
import random
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(1)

class CONFIG():
  dir_path = "../OAI_dataset_255/"
  save_path = "classification/knee/predictor_module/results/" 
  train_dir=dir_path + 'train' 
  val_dir=dir_path + 'val'

  device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_name = "predictor_module"
  folder_path = save_path + model_name
  save_folder = folder_path+ "/" + model_name  + ".pth"
  save_folder1 = folder_path+ "/" + model_name + "_loss.png"
  save_folder2 = folder_path+ "/" + model_name + "_cm.png"
  save_folder3 =folder_path + "/" + model_name + "_class_report.csv"
  save_folder4 =folder_path + "/" + model_name + "_Gradcam.png"
  save_folder5 =folder_path + "/" + model_name + "summary.txt"
  embeddings_path = folder_path + "/" + model_name + "embeddings.npy"
  pretrained = True

  num_classes = 5

  batch_size_train = 24
  batch_size_val = 32

  max_epochs = 50
  lr = 5e-4
  weight_decay = 5e-3
  lr_decay_epoch =5 

  num_workers = 1

cfg = CONFIG()

#import train
class_names0 = os.listdir(cfg.train_dir)
class_names = sorted(class_names0)
print(class_names)
num_class = len(class_names)
image_files = [[os.path.join(cfg.train_dir, class_name, x) 
               for x in os.listdir(os.path.join(cfg.train_dir, class_name))] 
               for class_name in class_names]

image_file_list = []
image_label_list = []
for i, class_name in enumerate(class_names):
    image_file_list.extend(image_files[i])
    image_label_list.extend([i] * len(image_files[i]))
    
#import valid
v_class_names0 = os.listdir(cfg.val_dir)
v_class_names = sorted(v_class_names0)
print(v_class_names)
v_num_class = len(v_class_names)
v_image_files = [[os.path.join(cfg.val_dir, v_class_name, x) 
               for x in os.listdir(os.path.join(cfg.val_dir, v_class_name))] 
               for v_class_name in v_class_names]

v_image_file_list = []
v_image_label_list = []
for i, class_name in enumerate(v_class_names):
    v_image_file_list.extend(v_image_files[i])
    v_image_label_list.extend([i]*len(v_image_files[i]))
    
#Save the file
k = pd.DataFrame(dict({"image_name":image_file_list}))
k.to_csv(cfg.folder_path+"/train_list.csv")
trainX=np.array(image_file_list)
trainY=np.array(image_label_list)
valX=np.array(v_image_file_list)
valY=np.array(v_image_label_list)

#Image Transformation
from monai.transforms import HistogramNormalize
import torchvision
pixel_mean, pixel_std = 0.66133188, 0.21229856

sharpen_filter=np.array([[-1,-1,-1],
                 [-1,9,-1],
                [-1,-1,-1]])
                
class MyResize(Transform):
    def __init__(self,size=(224,224)):
        self.size = size
    def __call__(self,inputs):
        sharp_image=cv2.filter2D(np.array(inputs),-1,sharpen_filter)
        image=cv2.resize(sharp_image,dsize=(self.size[1],self.size[0]),interpolation=cv2.INTER_CUBIC)        
        #image2=image[25:475,25:475]
        #smooth = cv2.GaussianBlur(image2,(3,3),0)
        return image #smooth
        
train_transforms= transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            MyResize(),
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)])

val_transforms= transforms.Compose([
            MyResize(),
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)])

#Five set of transformations:
#1. rotation
class RotateFixedAngle:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return F.rotate(img, self.angle)
        
rot_clockwise = transforms.Compose([
        RotateFixedAngle(15),  # 15° anticlockwise
        transforms.ToTensor()
        ])
rot_anticlock = transforms.Compose([
        RotateFixedAngle(-15),  # 15° anticlockwise
        transforms.ToTensor()
        ])

#2. Horizontal Flipping 
Flip_horizontal = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),  # Always apply horizontal flip
    transforms.ToTensor()
    ])


bright = transforms.Compose([
    transforms.ColorJitter(contrast=1.5),
    transforms.ToTensor()
])
#Contrast
contrast = transforms.Compose([
    transforms.ColorJitter(brightness=1.5),
    transforms.ToTensor()
])

            
class Xraydataset(Dataset):
     def __init__(self, image_file_list, label_file_list, transforms, t1, t2, t3, t4, t5):
         self.image_file_list = image_file_list
         self.label_file_list = label_file_list
         self.transforms = transforms
         self.t1 = t1
         self.t2 = t2
         self.t3 = t3
         self.t4 = t4
         self.t5 = t5
         
     def __len__(self):
         return len(self.image_file_list)
         
     def __getitem__(self,index):
          image = Image.open(self.image_file_list[index]).convert('RGB')
          return self.transforms(image), self.t1(image), self.t2(image), self.t3(image), self.t4(image), self.t5(image), self.label_file_list[index]
          
train_ds = Xraydataset(image_file_list, image_label_list, train_transforms, rot_clockwise, rot_anticlock, Flip_horizontal, bright, contrast)
train_loader = DataLoader(train_ds, batch_size = cfg.batch_size_train, shuffle =True, num_workers = cfg.num_workers)
val_ds = Xraydataset(v_image_file_list, v_image_label_list, val_transforms, rot_clockwise, rot_anticlock, Flip_horizontal, bright, contrast)
val_loader = DataLoader(val_ds, batch_size = cfg.batch_size_val, shuffle =False, num_workers = cfg.num_workers)

act = Activations(softmax=True)
to_onehot = AsDiscrete(to_onehot=num_class)# n_classes=num_class


from monai.networks.nets.densenet import DenseNet201
from torch.optim.lr_scheduler import StepLR
from monai.networks.nets.resnet import ResNet
from monai.networks.nets import SEResNet50,SEResNet101,HighResNet
import torch
from torchvision import datasets,transforms,models
import torch.nn as nn


#weights = torch.tensor([0.15,0.25,0.25,0.15,0.2],device=cfg.device)
loss_function = torch.nn.CrossEntropyLoss(weight=None) 
model1 = models.vgg19(pretrained=cfg.pretrained).to(cfg.device)

torch.autograd.set_detect_anomaly(True)

class VGGDETR(nn.Module):
    def __init__(self, model1, num_classes=cfg.num_classes):
        super(VGGDETR, self).__init__()
        self.vgg19 = model1
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        encoder_norm = nn.LayerNorm(512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6, norm=encoder_norm)
        self.row_embed = nn.Parameter(torch.rand(50, 512))
        self.col_embed = nn.Parameter(torch.rand(50,512))
        
        self.classifier = nn.Sequential(
            nn.Linear(2048//2, 2048//2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048//2, 1024//2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024//2, num_classes),
        ).to(cfg.device)

    def forward(self, x):
        x = self.vgg19.features(x)
        #print(x2.shape)
        x3 = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # Shape: (batch_size, hidden_dim, 1, 1)
        h, w = x3.shape[-2:]
        pos = torch.cat([
                         self.col_embed[:w].unsqueeze(0).repeat(h,1,1),
                         self.row_embed[:h].unsqueeze(0).repeat(1,w,1),
                         ], dim=1).flatten(0,1).unsqueeze(1)
                         
        x4 = self.transformer_encoder(pos+x3.flatten(2).permute(2,0,1))
        x5 = x4.permute(1,0,2)
        x5 = torch.flatten(x5, 1)
        x5 = self.classifier(x5)
        return x5, x
        
num_classes = cfg.num_classes  # Adjust this according to your problem
model = VGGDETR(model1, num_classes).to(cfg.device)

optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
val_interval = 1

#Loss Function
def weighted_loss(outputs, labels, args):
    softmax_op = torch.nn.Softmax(1)
    prob_pred = softmax_op(outputs)

    def set_weights():
        init_weights = np.array([[1, 3, 5, 7, 9],
                                 [3, 1, 3, 5, 7],
                                 [5, 3, 1, 3, 5],
                                 [7, 5, 3, 1, 3],
                                 [9, 7, 5, 3, 1]], dtype=np.float32)

        adjusted_weights = init_weights + 1.0
        np.fill_diagonal(adjusted_weights, 0)

        return adjusted_weights
    cls_weights = set_weights()

    batch_num, class_num = outputs.size()
    class_hot = np.zeros([batch_num, class_num], dtype=np.float32)
    labels_np = labels.data.cpu().numpy()
    for ind in range(batch_num):
        class_hot[ind, :] = cls_weights[labels_np[ind], :]
    class_hot = torch.from_numpy(class_hot)
    class_hot = torch.autograd.Variable(class_hot).cuda()

    loss = torch.sum((prob_pred * class_hot)**2) / batch_num

    return loss

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
                dist = F1.pairwise_distance(feat1.unsqueeze(0), feat2.unsqueeze(0))  # shape [1]
                #print(dist.size())
                sim = F1.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0))  # shape [1]
                #print(sim.size())
                sim = (sim + 1) / 2  # normalize to [0, 1]
                loss = dist * sim  # pull closer, weighted by current similarity
                pos_term += loss.item()

        # Negative pairs: near vs. far cluster
        if len(near_feats) > 0 and len(far_feats) > 0:
            for near_feat in near_feats:
                for far_feat in far_feats:
                    dist = F1.pairwise_distance(near_feat.unsqueeze(0), far_feat.unsqueeze(0))  # shape [1]
                    sim = F1.cosine_similarity(near_feat.unsqueeze(0), far_feat.unsqueeze(0))  # shape [1]
                    sim = (sim + 1) / 2  # normalize to [0, 1]
                    loss = dist * sim  # push apart, weighted by current similarity
                    neg_term += loss.item()

        total_loss += abs(pos_term - neg_term)

    return total_loss 


def audicor_loss(image_probs: torch.Tensor, transformed_probs: list, original_features: list, transformed_features: list, original_labels: list, alpha: float = 0.7):
    batch_size, num_classes = image_probs.shape

    # Compute cross-entropy loss between each image and its transformations
    loss_values = []
    image_indices = []

    for i, transform in enumerate(transformed_probs):
        loss = F1.cross_entropy(transform, image_probs, reduction='none')  # Compute loss per image
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
    final_loss = ((alpha * near_loss + (1 - alpha) * far_loss)+(contrastive_loss_value) / (5*cfg.batch_size_train))  # 
    #print(f"Final Loss: {final_loss}")
    return final_loss
    
#Scheduler

class LRScheduler():
    def __init__(self, init_lr=1.0e-4, lr_decay_epoch=10):
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch

    def __call__(self, optimizer, epoch):
        '''Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.'''
        lr = self.init_lr * (0.8 ** (epoch // self.lr_decay_epoch))
        lr = max(lr, 1e-8)
        if epoch % self.lr_decay_epoch == 0:
            print ('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return optimizer
          
#Training
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
epoch_loss_values_ev = list()
auc_metric = ROCAUCMetric()
metric_values = list()
metric_values_train = list()
lr_scheduler = LRScheduler(cfg.lr, cfg.lr_decay_epoch)
alpha = torch.nn.Parameter(torch.tensor(0.7))

smoothness_list = []
conv_rate_list = []
divergence_list = []
prev_loss = None  # for comparing with previous epoch


for epoch in range(cfg.max_epochs):
    print('-' * 10)
    print(f"epoch {epoch + 1}/{cfg.max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    label_list = []
    optimizer = lr_scheduler(optimizer, epoch)

    for i,batch_data in tqdm(enumerate(train_loader)):
        y_pred_train = torch.tensor([], dtype=torch.float32, device=cfg.device)
        y_train = torch.tensor([], dtype=torch.long, device=cfg.device)
        step += 1
        inputs,inputs1,inputs2,inputs3,inputs4,inputs5,labels = batch_data[0].to(cfg.device),batch_data[1].to(cfg.device), batch_data[2].to(cfg.device), batch_data[3].to(cfg.device), batch_data[4].to(cfg.device), batch_data[5].to(cfg.device), batch_data[6].to(cfg.device)
        optimizer.zero_grad()
        outputs, feat = model(inputs.float())     ##### 
        #print(feat.shape)
        o1, feat1 = model(inputs1.float())
        o2, feat2 = model(inputs2.float())
        o3, feat3 = model(inputs3.float())
        o4, feat4 = model(inputs4.float())
        o5, feat5 = model(inputs5.float())
        
        loss1 = weighted_loss(outputs, labels, cfg)
        loss2 = loss_function(outputs, labels)
        
        feat = torch.flatten(feat,1)
        ori_features = list(torch.unbind(feat, dim=0))
        label_list.append(labels)
        feat1 = torch.flatten(feat1, 1)
        tr1_f = list(torch.unbind(feat1, dim=0))
        feat2 = torch.flatten(feat2, 1)        
        tr2_f = list(torch.unbind(feat2, dim=0))
        feat3 = torch.flatten(feat3, 1) 
        tr3_f = list(torch.unbind(feat3, dim=0))
        feat4 = torch.flatten(feat4, 1)
        tr4_f = list(torch.unbind(feat4, dim=0))
        feat5 = torch.flatten(feat5, 1)
        tr5_f = list(torch.unbind(feat5, dim=0))       
        transformed_probs = [o1, o2, o3, o4, o5]
        transformed_features = [tr1_f, tr2_f, tr3_f, tr4_f, tr5_f]
        
        loss3 = audicor_loss(outputs, transformed_probs, ori_features, transformed_features, label_list, alpha)
        y_pred_train = torch.cat([y_pred_train, outputs], dim=0)
        y_train = torch.cat([y_train, labels], dim=0)
        #output1 = torch.max(outputs)
        #print(output1)
        #loss2 = lossL1(output1, labels)
        loss = loss1+loss2+loss3
        loss.backward()
        optimizer.step()
        #scheduler.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size
        if (epoch + 1) % 10 == 0:
              save_path = f"checkpoint_epoch_{epoch+1}.pth"
              torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': epoch_loss}, cfg.folder_path + "/" + save_path)
    #print(f"Checkpoint saved: {save_path}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")   
    y_onehot_train = [to_onehot(i) for i in y_train]
    y_pred_act_train = [act(i) for i in y_pred_train]
    auc_metric(y_pred_act_train, y_onehot_train)
    auc_result_train = auc_metric.aggregate()
    auc_metric.reset()
    del y_pred_act_train, y_onehot_train
    metric_values_train.append(auc_result_train)
    acc_value_train = torch.eq(y_pred_train.argmax(dim=1), y_train)
    acc_metric_train = acc_value_train.sum().item() / len(acc_value_train)        
    print(f" current accuracy: {acc_metric_train:.4f}")    
    
    if prev_loss is None:
        smoothness = 0
        conv_rate = 0
        divergence = 0
    else:
        smoothness = abs(epoch_loss - prev_loss)
        conv_rate = (prev_loss - epoch_loss) / prev_loss
        divergence = max(0, epoch_loss - prev_loss)  # positive = instability spike

    smoothness_list.append(smoothness)
    conv_rate_list.append(conv_rate)
    divergence_list.append(divergence)

    prev_loss = epoch_loss

    if (epoch + 1) % val_interval == 0:
        model.eval()
        epoch_loss_ev = 0
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=cfg.device)
            y = torch.tensor([], dtype=torch.long, device=cfg.device)
            for val_data in val_loader:
                val_images1,_,_,_,_,_, val_labels = val_data[0].to(cfg.device),val_data[1].to(cfg.device), val_data[2].to(cfg.device), val_data[3].to(cfg.device), val_data[4].to(cfg.device), val_data[5].to(cfg.device), val_data[6].to(cfg.device)
                outputs,_ = model(val_images1.float())
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                ev_loss = weighted_loss(outputs, val_labels, cfg)
                epoch_loss_ev += ev_loss.item()

            epoch_loss_values_ev.append(epoch_loss_ev)   
            y_onehot = [to_onehot(i) for i in y]
            y_pred_act = [act(i) for i in y_pred]
            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(auc_result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            
            if acc_metric > best_metric:
                best_metric = acc_metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), cfg.save_folder)
                print(f"saved new AUC Metric is: {best_metric_epoch}")
                
            print(f" current epoch: {epoch + 1} current AUC: {auc_result:.4f}"
                  f" current accuracy: {acc_metric:.4f}"
                  f" at epoch: {best_metric_epoch}")
            
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
dict1 = {"Loss_Values": epoch_loss_values}
F = pd.DataFrame(dict1)
F.to_csv(cfg.folder_path+"/"+"loss.csv")

dict2 = {"Loss_Values": epoch_loss_values_ev}
F = pd.DataFrame(dict2)
F.to_csv(cfg.folder_path+"/"+"ev_loss.csv")

#Visulization
plt.figure('train', (12,6))
plt.subplot(1,2,1)
plt.title("Epoch Average Loss")
x = [i+1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel('epoch')
plt.plot(x, y)
plt.subplot(1,2,2)
plt.title("Validation: Area under the ROC curve")
x = [val_interval * (i+1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel('epoch')
plt.plot(x,y)
plt.savefig(cfg.save_folder1)
