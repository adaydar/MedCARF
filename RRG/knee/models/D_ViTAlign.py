import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
#from monai.networks.blocks.unetr_block import UnetrUpBlock
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple
from torch import Tensor

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Bridger(nn.Module):

    def __init__(self,d_img=256,d_txt=50,d_model=256,nhead=8,num_stages=3,strides=2,num_layers=12,stage_id=1):
      super().__init__()
      self.d_img = d_img
      self.d_txt = d_txt
      self.d_model = d_model
      self.num_stages = num_stages
      self.num_layers = num_layers
      self.stage_id = stage_id

      self.fusion_v = Interactor(d_model=d_model, nhead=nhead)
      self.fusion_t = Interactor(d_model=d_model, nhead=nhead)
      self.zoom_in = nn.Conv2d(d_img, d_model, kernel_size=strides, stride=strides, bias=False)

      if self.stage_id == 4:
          self.zoom_out = nn.ConvTranspose2d(d_model, d_img, kernel_size=2, stride=2, output_padding=1, bias=False)
      else:
          self.zoom_out = nn.ConvTranspose2d(d_model, d_img, kernel_size=2, stride=2, output_padding=1, bias=False)

      self.linear1 = nn.Linear(d_txt, d_model)
      self.linear2 = nn.Linear(d_model, d_txt)
      self.ln_v = nn.LayerNorm(d_model)
      self.ln_t = nn.LayerNorm(d_model)

    def forward(self,device,vis,txt):
      #vis--->8,256,19,19
      vis = vis.to(device)
      txt = txt.to(device)
      txt = txt.permute(1,0,2)
      v = vis.clone()
      t = txt.clone()
      t = t.float()
      last_v,last_t = v,t
      v = self.zoom_in(v)
      t = self.linear1(t)
      B,C,H,W = v.shape
      v = v.reshape(B, C, -1).permute(2, 0, 1)
      v,t = self.ln_v(v), self.ln_t(t)
      cs_v,cs_t = self.fusion_v(v,t), self.fusion_t(t,v)
      cs_v=self.fusion_v(cs_v,cs_t)
      f_v = cs_v.permute(1, 2, 0).reshape(B, -1, H, W)
      fv = self.zoom_out(f_v)
      #vis = vis + v
      return fv


class Interactor(nn.Module):

    def __init__(self,d_model,nhead,dim_feedforward=128,dropout=0.1,activation="relu"):
      super().__init__()
      self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
      self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
      # Implementation of Feedforward model
      self.linear1 = nn.Linear(d_model, dim_feedforward)
      self.dropout = nn.Dropout(dropout)
      self.linear2 = nn.Linear(dim_feedforward, d_model)

      self.norm1 = nn.LayerNorm(d_model)
      self.norm2 = nn.LayerNorm(d_model)
      self.norm3 = nn.LayerNorm(d_model)
      self.dropout1 = nn.Dropout(dropout)
      self.dropout2 = nn.Dropout(dropout)
      self.dropout3 = nn.Dropout(dropout)

      self.activation = _get_activation_fn(activation)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
      return tensor if pos is None else tensor + pos

    def forward(self,tgt,memory,tgt_key_padding_mask:Optional[Tensor]=None,memory_key_padding_mask:Optional[Tensor]=None,pos:Optional[Tensor]=None,query_pos:Optional[Tensor]=None):
      # self attn
      q = k = self.with_pos_embed(tgt, query_pos)
      v = tgt
      tgt2 = self.self_attn(q, k, value=v, attn_mask=None,
                            key_padding_mask=tgt_key_padding_mask)[0] # [H*W, B, C]
      tgt = tgt + self.dropout1(tgt2)
      tgt = self.norm1(tgt)

      # cross attn
      tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                key=self.with_pos_embed(memory, pos),
                                value=memory, attn_mask=None,
                                key_padding_mask=memory_key_padding_mask)[0]
      tgt = tgt + self.dropout2(tgt2)
      tgt = self.norm2(tgt)

      # ffn
      tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
      tgt = tgt + self.dropout3(tgt2)
      tgt = self.norm3(tgt)
      return tgt

