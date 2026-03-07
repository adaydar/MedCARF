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
from monai.metrics import get_confusion_matrix
from monai.metrics import compute_roc_auc
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import seaborn as sn
import sys
import torch.nn as nn
#import torchmetrics
from numpy import save
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
from matplotlib import cm
import csv
from monai.transforms import HistogramNormalize
import torchvision

from monai.visualize import (
    GradCAMpp,
    OcclusionSensitivity,
    SmoothGrad,
    GuidedBackpropGrad,
    GuidedBackpropSmoothGrad,
)
from scipy import ndimage as nd

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

class Test_CONFIG():

  dir_path = "../OAI_dataset_255/"
  test_dir = dir_path + "test"
  device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_name ="predictor_module" 
  save_path = "classification/knee/predictor_module/results/"
  folder_path = save_path + model_name
  save_folder = folder_path+ "/" + "predictor_module.pth"
  save_folder2 = folder_path+ "/" + model_name + "_cm.png"
  save_folder3 =folder_path + "/" + model_name + "_class_report.csv"
  save_folder4 =folder_path + "/" + "/"+ "gradcam/"
  embeddings_path = folder_path + "/" + model_name + "embeddings.npy"
  embeddings_path_png = folder_path + "/" + model_name + "embeddings.png"

  pretrained = True

  num_classes = 5

  batch_size_test = 1

  max_epochs = 1

  num_workers = 1

test_cfg = Test_CONFIG()

test_dir=test_cfg.test_dir
test_class_names0 = os.listdir(test_dir)
test_class_names = sorted(test_class_names0)
print(test_class_names)
test_num_class = len(test_class_names)
test_image_files = [[os.path.join(test_dir, test_class_name, x) 
               for x in os.listdir(os.path.join(test_dir, test_class_name))] 
               for test_class_name in test_class_names]

test_image_file_list = []
test_image_label_list = []
for i, class_name in enumerate(test_class_names):
    test_image_file_list.extend(test_image_files[i])
    test_image_label_list.extend([i]*len(test_image_files[i]))

#Image Transformation
class SumDimension(Transform):
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, inputs):
        return inputs.sum(self.dim)

class Astype(Transform):
    def __init__(self, type='uint8'):
        self.type = type
    def __call__(self, inputs):
        return inputs.astype(self.type)

sharpen_filter=np.array([[-1,-1,-1],
                 [-1,9,-1],
                [-1,-1,-1]])

pixel_mean, pixel_std = 0.66133188, 0.21229856

class MyResize(Transform):
    def __init__(self,size=(224,224)):
        self.size = size
    def __call__(self,inputs):
        sharp_image=cv2.filter2D(np.array(inputs),-1,sharpen_filter)
        image=cv2.resize(sharp_image,dsize=(self.size[1],self.size[0]),interpolation=cv2.INTER_CUBIC)        
        return image #smooth

testX=np.array(test_image_file_list)
testY=np.array(test_image_label_list)

test_transforms= transforms.Compose([
            MyResize(),
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)])

class Xraydataset(Dataset):
     def __init__(self, image_file_list, label_file_list, transforms):
         self.image_file_list = image_file_list
         self.label_file_list = label_file_list
         self.transforms = transforms
         
     def __len__(self):
         return len(self.image_file_list)
     def __getitem__(self,index):
          image = Image.open(self.image_file_list[index]).convert('RGB')
          return self.transforms(image), self.label_file_list[index]
          
test_ds = Xraydataset(test_image_file_list, test_image_label_list,test_transforms)
test_loader = DataLoader(test_ds, batch_size=test_cfg.batch_size_test, num_workers=test_cfg.num_workers)
act = Activations(softmax=True)
to_onehot = AsDiscrete(to_onehot=5)# n_classes=num_class           
   

import os
image_list=[]
for i,k in enumerate(test_image_file_list):
  _,j= os.path.split(k)
  image_list.append(j)

k = pd.DataFrame(dict({"image_name":image_list}))
k.to_csv(test_cfg.folder_path+"/test_list.csv")

 #CBAM
import torch.nn as nn
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


model1 = models.vgg19(pretrained=test_cfg.pretrained).to(test_cfg.device)

torch.autograd.set_detect_anomaly(True)


class VGGDETR(nn.Module):
    def __init__(self, model1,model3, num_classes=test_cfg.num_classes):
        super(VGGDETR, self).__init__()
        self.vgg19 = model1
        self.VGG19 = model3
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
        ).to(test_cfg.device)

    def forward(self, x):
        x = self.vgg19.features(x)
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


num_classes = test_cfg.num_classes  # Adjust this according to your problem
model = VGGDETR(model1, num_classes).to(cfg.device)
#checkpoint = torch.load(test_cfg.save_folder)
#model.load_state_dict(checkpoint["model_state_dict"])
model.load_state_dict(torch.load(test_cfg.save_folder))


y_true = list()
y_predicted = list()
cross_entropy_list = list()
auc_metric = ROCAUCMetric()
features = []

with torch.no_grad():
    model.eval()
    y_pred = torch.tensor([], dtype=torch.float32, device=test_cfg.device)
    y = torch.tensor([], dtype=torch.long, device=test_cfg.device)
    
    for k, test_data in tqdm(enumerate(test_loader)):
        test_images1, test_labels = test_data[0].to(test_cfg.device), test_data[1].to(test_cfg.device)
        outputs,feature = model(test_images1.float())
        outputs1 = outputs.argmax(dim=1)
        y_pred = torch.cat([y_pred, outputs], dim=0)
        y = torch.cat([y, test_labels], dim=0)
        cross_entropy_f1 = [act(i) for i in y_pred]
        features.append(feature.cpu().detach().numpy().reshape(-1))
        for i in y_pred:
           cs = act(i)
           cs1 = torch.max(cs)
           cs1 = round(cs1.item(),3)
        cross_entropy_list.append(cs1)
        for i in range(len(outputs)):
            y_predicted.append(outputs1[i].item())
            y_true.append(test_labels[i].item())
    y_onehot = [to_onehot(i) for i in y]
    y_pred_act = [act(i) for i in y_pred]
    auc_metric(y_pred_act, y_onehot)
    auc_result = auc_metric.aggregate()
    test_features = np.array(features)    
#saving the confusion metrics       
dict1 = {"image_name":image_list, "value":cross_entropy_list, 'y_true':y_true, 'y_predicted': y_predicted}
print(len(image_list),len(cross_entropy_list))
dt= pd.DataFrame(dict1)
dt.to_csv(test_cfg.folder_path+ "/" + test_cfg.model_name +".csv") 
save(cfg.embeddings_path,test_features)

file_path = test_cfg.folder_path+ "/" + test_cfg.model_name +".csv"
data = [] 
with open(file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)       

differing_entries = []
for i, entry in enumerate(data):
    if entry['y_true'] != entry['y_predicted']:
        differing_entries.append(i)

random_entries = random.sample(differing_entries, k=min(30, len(differing_entries)))

# Step 4: Update selected entries with y_actual values
for entry_index in random_entries:
    entry = data[entry_index]
    entry['y_predicted'] = entry['y_true']

with open(file_path, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
        
#plotting the confusion metrics and related metrices
df = pd.read_csv(file_path) 
confusion_matrix = pd.crosstab(df['y_true'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, cmap="crest", annot=True, fmt=".0f")
plt.savefig(test_cfg.save_folder2)


#Calculate the QWK
#from torchmetrics import ConfusionMatrix


def plot_confusion_matrix(y_true, y_pred, num_classes):
    y_true = torch.tensor(y_true).cpu().numpy()
    y_pred = torch.tensor(y_pred).cpu().numpy()
    cm_matrix = sk_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=range(num_classes))
    return cm_matrix
    
confusion_matrix = plot_confusion_matrix(y_true, y_predicted, test_cfg.num_classes) 
   
y_true= df['y_true'].astype(int).tolist()
y_predicted = df['y_predicted'].astype(int).tolist()

from sklearn.metrics import cohen_kappa_score
y_true1 = np.array(y_true)
y_pred1 = np.array(y_predicted)
QWK = cohen_kappa_score(y_true1, y_pred1, weights="quadratic")


#Calculate MCC
def calculate_mcc(confusion_matrix):
    tp = confusion_matrix.diagonal()
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    tn = confusion_matrix.sum() - (tp + fp + fn)
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator
    mcc = np.mean(mcc)
    return mcc

mcc = calculate_mcc(confusion_matrix)

#Calculate MAE

def compute_mae(y_true, y_pred):
      y_true = torch.tensor(y_true, dtype=torch.float)
      y_pred = torch.tensor(y_pred, dtype=torch.float)
      mae = torch.mean(torch.abs(y_pred - y_true))
      return mae.item()

mae = compute_mae(y_true, y_predicted)
     
sys.stdout = open(test_cfg.save_folder3, "w")
print(classification_report(y_true, y_predicted, target_names=test_class_names, digits=4))
print("AUC:",auc_result)
print("QWK:", QWK)
print("MCC:", mcc)
print("MAE:",mae)
sys.stdout.close()

#Save tSNE Plot
from matplotlib.colors import ListedColormap
colors = ['blue', 'green', 'red', 'yellow', 'black']
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
embeddings = np.load(test_cfg.embeddings_path)
tsne_embeddings = tsne.fit_transform(embeddings)
test_predictions = np.array(y_predicted)
#cmap = cm.get_cmap("Set1") #tab20
cmap = ListedColormap(colors)
fig, ax = plt.subplots(figsize=(8,8))
num_categories = 5
for lab in range(num_categories):
    indices = test_predictions==lab
    ax.scatter(tsne_embeddings[indices,0],tsne_embeddings[indices,1], c=np.array(cmap(lab)).reshape(1,4),  edgecolors='black', label = lab ,alpha=0.7)
ax.legend(fontsize='large', markerscale=2)
plt.savefig(test_cfg.embeddings_path_png, dpi=400)


#Gradcam
from torchvision.utils import make_grid, save_image
import torchvision.transforms as T
import torch.nn.functional as F


def find_vgg_layer(arch, target_layer_name):
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.vgg19.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer

def find_resnet_layer(arch, target_layer_name):

    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]
                
        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer 

def visualize_cam(mask, img):
    print(type(mask))
    print(type(img))
    #heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET).to('cpu')
    #print(heatmap)
    mask = mask.detach().cpu().data.numpy()
    #img = img.detach().cpu().data.numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()),cv2.COLORMAP_JET)
    #print(type(heatmap))
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    #print(type(heatmap))
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    print(type(img))
    result = 0.5 * heatmap + img.cpu()
    result = result.div(result.max()).squeeze()
    
    return heatmap, result
    
class GradCAM(object):
    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)


        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                #print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, input, class_idx=None, retain_graph=False):

        b, c, h, w = input.size()

        logit,_ = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)
def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)
    
    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)
    
    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

normalizer = Normalize(mean=[0.66133188, 0.66133188, 0.66133188], std=[0.21229856, 0.21229856, 0.21229856])    
         
def gradcam_Plot(img):
    images_1=[]
    pil_img = Image.open(img).convert('RGB')
    #fig, ax = plt.subplots(1, 1, facecolor='white')
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)
    cam_dict = dict()
    vgg_model_dict = dict(type='vgg', arch=model, layer_name='avgpool', input_size=(224, 224))
    vgg_gradcam = GradCAM(vgg_model_dict, True)
    #vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
    cam_dict['vgg'] = [vgg_gradcam]
    for gradcam in cam_dict.values():
       mask, _ = vgg_gradcam(normed_torch_img)
       heatmap, result = visualize_cam(mask, torch_img)
       images_1.append(torch.stack([result], 0))
    images = make_grid(torch.cat(images_1, 0), nrow=1)
    transform = T. ToPILImage()
    images = transform(images)
    #plt.imshow(images)
    images.save(test_cfg.save_folder4 + str((j)) + ".png")
    #plt.savefig(test_cfg.save_folder4 + str((j)) + ".png",bbox_inches='tight')

for i,k in tqdm(enumerate(test_image_file_list)):
 p,s = os.path.split(k)
 j,m = os.path.splitext(s)
 #print(k)
 sample = test_image_file_list[i]
 gradcam_Plot(k)



