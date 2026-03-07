import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import util
import os
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt, fn = util.parser_model('test')

model_path = fn+'/model_auc.pkl'

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.gradients = None
        self.activations = None

        def fwd_hook(module, inp, out):
            #print("Forward hook triggered")
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_backward_hook(bwd_hook)

    def generate(self, input_tensor, target_index=None):
        self.model.zero_grad()
        #print(input_tensor.shape)
        x = self.model.model.model(input_tensor)  
        #model.set_input(test_data)
        #print(x.shape)
        o1 = nn.ReLU(inplace=True)(x)
        x3 = nn.AdaptiveAvgPool2d((1, 1))(o1)
        x4 = x3.view(x3.size(0), -1)
        #print(x4.shape)
        #print(x4.device)
        fc = nn.Linear(1024, 15).to(device)  # move Linear to same device
        output = torch.sigmoid(fc(x4))
        #print(output)

        # Multi-label (sigmoid) OR multi-class (softmax)
        if target_index is None:
            target_index = output.argmax(dim=1).item()

        score = output[:, target_index]
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear')
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, target_index
        

def overlay_heatmap(img_tensor, heatmap, alpha=0.5):
    """
    img_tensor: CHW tensor normalized with ImageNet stats
    heatmap: 2D numpy float heatmap from GradCAM
    """

    # 1. Denormalize the image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img_tensor.cpu() * std + mean  # put back into [0,1]
    img = (img.clamp(0,1).permute(1,2,0).numpy() * 255).astype("uint8")

    # 2. Normalize heatmap to [0,1]
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # 3. Resize to match image shape
    hmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 4. Apply colormap
    hmap = np.uint8(255 * hmap)
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)

    # 5. Convert BGR → RGB
    hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)

    # 6. Alpha blend
    overlay = cv2.addWeighted(img, 1 - alpha, hmap, alpha, 0)

    # 7. Return PIL image
    return Image.fromarray(overlay)

def blur_background(img_tensor, heatmap, blur_strength=25, threshold=0.4):
    """
    Keeps areas with high activation sharp, blurs all other parts.
    img_tensor: CHW normalized tensor
    heatmap: 2D float array (0-1)
    """

    # ---- 1. Reconstruct RGB image (same as overlay function) ----
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img_tensor.cpu() * std + mean
    img = (img.clamp(0,1).permute(1,2,0).numpy() * 255).astype("uint8")

    # ---- 2. Normalize heatmap ----
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # ---- 3. Resize heatmap to original size ----
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # ---- 4. Build binary mask (activation area = 1) ----
    mask = (heatmap >= threshold).astype("float32")   # threshold e.g. 0.4
    mask_3c = np.stack([mask]*3, axis=-1)

    # ---- 5. Blur full image ----
    blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)

    # ---- 6. Composite: sharp where mask=1, blurred where mask=0 ----
    final = (img * mask_3c + blurred * (1 - mask_3c)).astype("uint8")

    return Image.fromarray(final)
    
def generate_gradcam(dataloader, model, target_layer, save_dir="gradcam_outputs", save_dir1 = "images_seg", target_class=None):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir1, exist_ok = True)
    gradcam = GradCAM(model, target_layer)
    device = next(model.parameters()).device

    for batch in dataloader:
        images = batch['im'].to(device)            # (B, 3, 224, 224)
        img_ids = batch['img_path']                # list of ids for saving

        for i in range(images.size(0)):
            img = images[i:i+1]

            heatmap, cls = gradcam.generate(img, target_index=target_class)

            overlay = overlay_heatmap(images[i].cpu(), heatmap)

            filename = f"{img_ids[i]}_gradcam_cls{cls}.png"
            overlay.save(os.path.join(save_dir, filename))
            clear_focus = blur_background(images[i], heatmap)
            fname2 = f"{img_ids[i]}_focused_cls{cls}.png"
            clear_focus.save(os.path.join(save_dir1, fname2))

            print(f"Saved: {fname2}")

image_size = opt.image_size
batch_size = opt.batch_size
class_name = opt.class_name
test_loader, test_dataset_size = util.load_data('file/test.csv', 'test', batch_size, image_size, class_name)

model = torch.load(model_path, weights_only=False)  # your trained model
#print(model)
#for name, param in model.named_parameters():
#    print(name, param.requires_grad)
#print(model.model.model.denseblock4.denselayer16.conv2)
target_layer = model.model.model.denseblock4.denselayer16.conv2 #model.model.denseblock4.denselayer16.conv2 # e.g., Densenet121 last conv block 
#print(target_layer)

generate_gradcam(test_loader, model, target_layer, save_dir="gradcam_auto")
