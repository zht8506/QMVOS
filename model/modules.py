"""
modules.py - This file stores the rather boring network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask. 
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.group_modules import *
from model import resnet
from model.cbam import CBAM
from model.attention_block import MLP,Self_Cross_FFN
import math


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 64,
    temperature: int = 10000,
    exchange_xy: bool = True,) -> torch.Tensor:
    """generate sine position embedding from a position tensor

    Args:
        pos_tensor (torch.Tensor): Shape as `(None, n)`.
        num_pos_feats (int): projected shape for each float in the tensor. Default: 128
        temperature (int): The temperature used for scaling
            the position embedding. Default: 10000.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is `[x, y]`, the results will  # noqa 
            be `[pos(y), pos(x)]`. Defaults: True.

    Returns:
        torch.Tensor: Returned position embedding  # noqa 
        with shape `(None, n * num_pos_feats)`.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=2)
    return pos_res

class SimpleUpsampleBlock(nn.Module):
    def __init__(self, lr_in_channel, hr_in_channel, mid_channel, out_channel, is_deconv=False):
        super().__init__()
        self.output_conv = nn.Sequential(
            nn.Conv2d(mid_channel+hr_in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False,),
            nn.GroupNorm(32,  out_channel),
            #act_layer(),
            )
        if is_deconv:
            self.up = nn.ConvTranspose2d(lr_in_channel, mid_channel, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                 nn.UpsamplingBilinear2d(scale_factor=2),
                 nn.Conv2d(lr_in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False,),
                 nn.GroupNorm(32,  mid_channel),
                )
        
    def forward(self, lr_feat, hr_feat):
        # if need add frame dim
        if len(lr_feat.shape)==4:
            lr_feat=lr_feat.unsqueeze(1)
        if len(hr_feat.shape)==4:
            hr_feat=hr_feat.unsqueeze(1)
        
        b, t, dim, _, _ = lr_feat.shape
        
        lr_feat = self.up( lr_feat.flatten(start_dim=0, end_dim=1) ) #(bs*t, dim, hr_h, hr_w)
        hr_feat = hr_feat.flatten(start_dim=0, end_dim=1) #(bs*t, dim, hr_h, hr_w)
        cat_feat = torch.cat( (lr_feat, hr_feat), 1) 
        
        output = self.output_conv(cat_feat)
        output = output.view(b, t, dim, *output.shape[-2:])
        
        return output

class SimpleFusion(nn.Module):
    def __init__(self, f4_channel,f8_channel, f16_channel, out_channel):
        super().__init__()
        
        self.out_channel=out_channel

        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channel*2+f8_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False,),
            nn.GroupNorm(32,  out_channel),
            #act_layer(),
            )
        
        self.down = nn.Sequential(
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(f4_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False,),
                nn.GroupNorm(32,  out_channel),
            )

        self.up = nn.Sequential(
                 nn.UpsamplingBilinear2d(scale_factor=2),
                 nn.Conv2d(f16_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False,),
                 nn.GroupNorm(32,  out_channel),
                )
        
      
    def forward(self, f4,f8,f16):
        # if need add frame dim
        if len(f4.shape)==4:
            f4=f4.unsqueeze(1)
        if len(f8.shape)==4:
            f8=f8.unsqueeze(1)
        if len(f16.shape)==4:
            f16=f16.unsqueeze(1)
        
        b, t, _, _, _ = f8.shape
        dim=self.out_channel

        f4_down = self.down(f4.flatten(start_dim=0, end_dim=1))
        f16_up  = self.up(f16.flatten(start_dim=0, end_dim=1))
        
        f8 = f8.flatten(start_dim=0, end_dim=1) #(bs*t, dim, hr_h, hr_w)
        cat_feat = torch.cat( (f4_down, f8, f16_up), 1) 
        
        output = self.output_conv(cat_feat)
        output = output.view(b, t, dim, *output.shape[-2:])
        
        return output

def Weighted_GAP(supp_feat, mask):
    #supp_feat with shape (bs, num_obj, embed_dim, h, w)
    #mask with shape      (bs, num_obj,         1, h, w) 
    
    B, num_obj, embed_dim, feat_h, feat_w = supp_feat.shape
    
    #number non-zero pixels in mask
    area = F.avg_pool2d(input=mask.flatten(start_dim=0, end_dim=1), kernel_size=(feat_h, feat_w)) * feat_h * feat_w + 0.0005  #(bs*num_obj, 1, 1, 1)
    
    #masked feature
    masked_feat = supp_feat * mask #(bs, num_obj, embed_dim, h, w)
    
    #sum-up all masked features, multiply feat_h*feat_w due to averaging of F.avg_pool2d
    supp_feat = F.avg_pool2d(input=masked_feat.flatten(start_dim=0, end_dim=1), kernel_size=(feat_h,feat_w)) * feat_h * feat_w #(bs*num_obj, embed_dim, 1, 1)
    
    query_embed = supp_feat/area #(bs*num_obj, embed_dimm, 1, 1)
    query_embed = query_embed.view(B, num_obj, embed_dim) #(bs, num_obj, embed_dim)
    
    return query_embed

def prob2mask(prob_w_bg):

    bs,num_objects_bg=prob_w_bg.shape[:2]
    num_objects=num_objects_bg-1

    mask_prev=torch.argmax(prob_w_bg, dim=1)
    mask_prev=[mask_prev==i for i in range(1,num_objects+1)]
    mask_prev=[i.int() for i in mask_prev]
    mask_prev=torch.stack(mask_prev,dim=1)

    mask_prev=mask_prev.reshape(bs,num_objects,*mask_prev.shape[-2:])
    return mask_prev

class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]

        g = self.distributor(x, g)
        g = self.block1(g)
        r = self.attention(g.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])

        g = self.block2(g+r)

        return g


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dims, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.g16_conv = GConv2D(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2D(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2D(g_dims[2], mid_dim, kernel_size=1)

        self.transform = GConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
            self.g4_conv(downsample_groups(g[2], ratio=1/4))

        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = GConv2D(g_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class ValueEncoder(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False):
        super().__init__()
        
        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool

        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)

        g = self.conv1(g)
        g = self.bn1(g)
        g = self.maxpool(g)
        g = self.relu(g) 

        g = self.layer1(g)
        g = self.layer2(g)
        g = self.layer3(g)

        g = g.view(batch_size, num_objects, *g.shape[1:])
        g = self.fuser(image_feat_f16, g)

        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)

        return g, h
 

class fpn(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dims):
        super().__init__()

        self.g4_conv = GConv2D(g_dims[0], g_dims[0], kernel_size=3,stride=2,padding=1)
        self.g8_conv = GConv2D(g_dims[1]+g_dims[0], g_dims[1]+g_dims[0], kernel_size=3,stride=2,padding=1)
        self.g16_conv = GConv2D(g_dims[1]+g_dims[0]+g_dims[2], g_dims[2], kernel_size=3,stride=1,padding=1)

    def forward(self, multi_feature):
        
        g4= self.g4_conv(multi_feature[0])
        g8=torch.cat([multi_feature[1],g4],dim=1)
        g8=self.g8_conv(g8)
        g16=torch.cat([multi_feature[2],g8],dim=1)

        g16=self.g16_conv(g16)

        return g16

class ValueEncoder_multi_scale(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False):
        super().__init__()
        
        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 64
        self.layer2 = network.layer2 # 1/8, 128
        self.layer3 = network.layer3 # 1/16, 256

        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

        self.fpn=fpn((64,128,256))

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)

        g = self.conv1(g)
        g = self.bn1(g) # 1/2, 64
        g = self.maxpool(g)  # 1/4, 64
        g = self.relu(g) 

        g4 = self.layer1(g) # 1/4, [bs,64,96,96]
        g8 = self.layer2(g4) # 1/8 [bs,128,48,48]
        g16 = self.layer3(g8) # 1/16 [bs,256,24,24]

        g_multi_scale=self.fpn((g4,g8,g16))

        g_multi_scale = g_multi_scale.view(batch_size, num_objects, *g_multi_scale.shape[1:])
        g_multi_scale = self.fuser(image_feat_f16, g_multi_scale)

        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g_multi_scale, h)

        return g_multi_scale, h
 

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.res2 = network.layer1 # 1/4, 256
        self.layer2 = network.layer2 # 1/8, 512
        self.layer3 = network.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g):
        skip_f = self.skip_conv(skip_f)
        g = upsample_groups(up_g, ratio=self.scale_factor)
        g = self.distributor(skip_f, g)
        g = self.out_conv(g)
        return g


class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim):
        super().__init__()

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        # shrinkage
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x, need_s, need_e):
        shrinkage = self.d_proj(x)**2 + 1 if (need_s) else None
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None

        return self.key_proj(x), shrinkage, selection


class Decoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()

        self.fuser = FeatureFusionBlock(1024, val_dim+hidden_dim, 512, 512)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim)
        else:
            self.hidden_update = None
        
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
        batch_size, num_objects = memory_readout.shape[:2]

        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)

        g8 = self.up_16_8(f8, g16)
        g4 = self.up_8_4(f4, g8)
        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))

        if h_out and self.hidden_update is not None:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        else:
            hidden_state = None
        
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return hidden_state, logits



class Decoder_QCIM(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()

        self.fuser = FeatureFusionBlock(1024, val_dim+hidden_dim, 512, 512)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim)
        else:
            self.hidden_update = None
        
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4
        self.g16_proj=GConv2D(512,256,1,1)

        self.QCIM=Self_Cross_FFN(d_model=256)

    def forward(self, query_embed,num_filled_objects,f16, f8, f4, hidden_state, memory_readout, h_out=True):
        batch_size, num_objects = memory_readout.shape[:2]

        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)

        #-------------------------Query-Content Interaction Module-------------------------------#
        g16_proj=self.g16_proj(g16).permute(0,2,1,3,4).flatten(2).transpose(-2,-1) # [B,512,num_onjects*H*W]

        # generate mask
        maskSelf = torch.zeros(batch_size, num_objects, num_objects,device=g16_proj.device)
        for i, valid_queries in enumerate(num_filled_objects):
            maskSelf[i, :, valid_queries:] = -1e9
        maskSelf = maskSelf.unsqueeze(1).expand(-1,8,-1,-1).flatten(0,1)
        
        num_heads = 8
        img_area = g16.shape[-2]*g16.shape[-1]
        cross_mask = torch.ones(batch_size*num_heads, num_objects, num_objects*img_area, device=g16_proj.device)
        for i, valid_queries in enumerate(num_filled_objects):
            cross_mask[i*num_heads:(i+1)*num_heads, :, :valid_queries*img_area] = 0
        cross_mask = cross_mask.bool()

        # self-attention + cross-attention + ffn
        query_embed=self.QCIM(tgt=query_embed, query_pos=None, 
                              src=g16_proj, self_attn_mask=maskSelf, 
                              cross_attn_mask=cross_mask)
        #---------------------------------------------------------------#

        g8 = self.up_16_8(f8, g16)
        g4 = self.up_8_4(f4, g8) # [4,3,256,96,96]

        logits=[]
        for i in range(num_objects):
            query_tmp=query_embed[:,i,:]
            g4_tmp=g4[:,i,:,:,:]
            logits_tmp = torch.einsum("bc,bchw->bhw", query_tmp, g4_tmp)
            logits.append(logits_tmp.unsqueeze(1))
        logits=torch.cat(logits,dim=1)
        logits=logits.reshape(batch_size*num_objects,1,*g4.shape[-2:])

        if h_out and self.hidden_update is not None:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        else:
            hidden_state = None
        
        # interpolate upsamle 4X
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return hidden_state, logits

