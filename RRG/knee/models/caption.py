import torch
from torch import nn
import torch.nn.functional as F
from .utils import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer
from .D_VitAlign import Bridger

class Caption(nn.Module):
    def __init__(self, backbone, transformer, bridger, hidden_dim, vocab_size):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(
            backbone.num_channels, hidden_dim, kernel_size=1
        )
        self.transformer = transformer
        self.bridger = bridger
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)
        self.TabularMLP = TabularMLP(12).to('cuda')

    def forward(self, samples, target, target_mask, class_feature, tab_features, graph_features):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        src_proj = self.input_proj(src)

        att_features = self.bridger(target.device, src_proj, graph_features)

        hs = self.transformer(
            att_features, mask, pos[-1], target, target_mask, class_feature
        )
        #hs = self.transformer(self.input_proj(src), mask, pos[-1], target, target_mask, class_feature)
        #print(hs.size())
        # Pass Transformer output through MLP for final predictions
        out = self.mlp(hs.permute(1, 0, 2))
        #print(out.size())
        tab_features = tab_features.to('cuda')
        out1 = self.TabularMLP(tab_features)
        out1 = out1.unsqueeze(1).expand(-1,128,-1)
        #extra_feature_proj = self.extra_feature_projection(extra_feature)  # Align dimensions
        #out1 = out1.permute(1, 0, 2) 
        #print(out1.size())
        #print(out.size())
        #print(out1.type())
        #print(out.type())
        out_cat = torch.cat([out,out1], dim=2)
        return out_cat

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 64], output_dim=128):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

def build_model(config):
    backbone = build_backbone(config)
    transformer = build_transformer(config)
    bridger = Bridger()

    model = Caption(backbone, transformer, bridger, config.hidden_dim, config.vocab_size)
    criterion = torch.nn.CrossEntropyLoss()

    return model, criterion

