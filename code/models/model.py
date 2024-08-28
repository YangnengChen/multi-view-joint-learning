###--------- This code refers to https://github.com/pengzhiliang/Conformer ------###

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.non_local_simple_version import NONLocalBlock2D
from timm.models.layers import DropPath, trunc_normal_
from collections import OrderedDict
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from ml_decoder import MLDecoder
import argparse
import datetime
import json
from functools import reduce
import numpy as np
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from ml_decoder import MLDecoder
import timm
from timm.models.nfnet import dm_nfnet_f5
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler


from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        print('global_pool:',self.global_pool)
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    # def forward_features(self, x):
    #     B = x.shape[0]
    #     x = self.patch_embed(x)

    #     cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x = x + self.pos_embed
    #     x = self.pos_drop(x)

    #     for blk in self.blocks:
    #         x = blk(x)

    #     if self.global_pool:
    #         x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
    #         outcome = self.fc_norm(x)
    #     else:
    #         x = self.norm(x)
    #         outcome = x[:, 0]
        
    #     return outcome
          
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
            
        # outcome=x

        return outcome


def backbone_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

class Nfnet(nn.Module):
    def __init__(self, n_classes, model_path,pretrained=False,weight_type='npz',embed_dim=None,freeze_layers=False,using_decoder=False):

        super(Nfnet, self).__init__()

        self.model =dm_nfnet_f5()
        


        if pretrained:
            if weight_type=='npz':
                self.model.load_pretrained(checkpoint_path=model_path)
            elif weight_type == 'pth':
                if model_path != "":
                    assert os.path.exists(model_path), "weights file: '{}' not exist.".format(model_path)
                    weights_dict = torch.load(model_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                    # weights_dict = torch.load(model_path, map_location=torch.device( 'cpu'))
                    if isinstance(weights_dict, OrderedDict):
                        for key in list(weights_dict.keys()):
                            if key.startswith('classifier.'):
                                weights_dict.pop(key)
                        print(self.model.load_state_dict(weights_dict, strict=False))
                    elif isinstance(weights_dict, dict):
                        if 'model' in weights_dict:
                            print(self.model.load_state_dict(weights_dict['model'], strict=False))
                        else:
                            if  'blocks.0.mlp.w12.weight' in weights_dict:
                                for block_idx in range(40):
                                # 生成旧键和新键
                                    old_weight_keys = [
                                        f'blocks.{block_idx}.mlp.w12.weight',
                                        f'blocks.{block_idx}.mlp.w12.bias',
                                        f'blocks.{block_idx}.mlp.w3.weight',
                                        f'blocks.{block_idx}.mlp.w3.bias'
                                    ]
                                    
                                    new_weight_keys = [
                                        f'blocks.{block_idx}.mlp.fc1.weight',
                                        f'blocks.{block_idx}.mlp.fc1.bias',
                                        f'blocks.{block_idx}.mlp.fc2.weight',
                                        f'blocks.{block_idx}.mlp.fc2.bias'
                                    ]

                                    # 遍历旧键和新键
                                    for old_key, new_key in zip(old_weight_keys, new_weight_keys):
                                        # 使用旧键从 weight_dict 中获取权重张量
                                        weight_tensor = weights_dict[old_key]
                                        # 删除旧键
                                        del weights_dict[old_key]
                                        # 使用新键将权重张量添加回 weight_dict 中
                                        weights_dict[new_key] = weight_tensor

                                # 现在，weight_dict 中包含了更新后的权重键
                                print(self.model.load_state_dict(weights_dict, strict=False))   


                            else:
                                print(self.model.load_state_dict(weights_dict, strict=False))
                    else:
                        print('Error')
        
        self.model.head.fc = nn.Identity()
            
        if freeze_layers:
            for name, params in self.model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    # print(name)
                    params.requires_grad_(False)
                else:
                    print("training {}".format(name))  
            
    # @autocast()    
    def forward(self, x):
        x = self.model(x)
        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def load_checkpoint(model,checkpoint_path,global_pool=True,device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % checkpoint_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    # 位置嵌入插值以适应高分辨率
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # if global_pool:
    #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    # else:
    #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)

    # model.to(device)

class SKNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        """
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        """
        super(SKNet, self).__init__()
        d = max(in_channels // r, L)  
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList() 
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1) 
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)  
        self.softmax = nn.Softmax(dim=1) 
    def forward(self, input):
        batch_size = input.size(0)
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))# output 为不同路径的卷积得到的结果
        U = reduce(lambda x, y: x + y, output)  
        s = self.global_pool(U)  
        z = self.fc1(s)
        a_b = self.fc2(z) 
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1) 
        a_b = self.softmax(a_b) 
        a_b = list(a_b.chunk(self.M, dim=1))  
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1),
                       a_b))  
        V = list(map(lambda x, y: x * y, output,
                     a_b))  # a_b为不同路径计算得到的注意力权重，乘以output再相加得到不同路径注意力加权的结果
        V = reduce(lambda x, y: x + y,
                   V)  
        return V



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):


    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)
        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)
        return x


class FCUUp(nn.Module):


    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class Med_ConvBlock(nn.Module):

    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):


    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion
    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape
        x_t = self.trans_block(x_t)
        x_st = self.squeeze_block(x2, x_t)


        x_tout = x_t + x_st
        # if no fusion for transformer branch, use it!
        # x_tout = x_t
        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        # if no fusion for convolution branch, mask it!
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_tout

class MergeView(nn.Module):
    def __init__(self, in_features, num_views):
        super(MergeView, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_views))  # 可学习的权重参数
        self.num_views = num_views
        
    def forward(self, x):
        # 将四个视图的特征乘以对应的权重并求和
        weighted_features = torch.stack([(x[:, i, :, :, :] * self.weight[i]) for i in range(self.num_views)], dim=1)
        fused_feature = torch.sum(weighted_features, dim=1)
        return fused_feature

class MergeView_albation(nn.Module):
    def __init__(self, in_features, num_views):
        super(MergeView_albation, self).__init__()
        
        
    def forward(self, x):
        # 将四个视图的特征取平均值
        avg_feature = torch.mean(x, dim=1)
        return avg_feature


class FuseViewFeatures(nn.Module):
    def __init__(self, in_channels, out_channels,SK_net=False ,M=2, r=16, L=32):
        super(FuseViewFeatures, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.using_SK_net=SK_net
        if SK_net:
            self.SK_net=SKNet(in_channels=4*3072,out_channels=4*3072,M=M,r=r,L=L)
    def forward(self, x):
        # 输入x的形状为[batch_size, num_views, embedding, H, W]
        # 将视图维度移到通道维度，得到形状为[batch_size, embedding*num_views, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(x.size(0), -1, x.size(3), x.size(4))
        if self.using_SK_net:
            x=self.SK_net(x)
        # 使用卷积层将四个视图的特征融合为[batch_size, embedding, H, W]
        fused_feature = self.conv(x)
        fused_feature = self.relu(fused_feature)

        return fused_feature
    
class FuseView(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FuseView, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)

    def joint(self,x):
        # x= rearrange(x,'(b v) c h w -> b v c h w',v=4)
        b ,v ,c ,h ,w=x.shape
        arr = x.clone()
        arr = arr.view(b,c,2*h,2*w)
        arr[:,:,:h,:w] = x[:,0,:,:,:]
        arr[:,:,:h,w:2*w] = x[:,1,:,:,:]
        arr[:,:,h:2*h,:w] = x[:,2,:,:,:]
        arr[:,:,h:2*h,w:2*w] = x[:,3,:,:,:]
        return arr
    
    def forward(self, x):
        x=self.joint(x)
        # 使用卷积层将四个视图的特征融合为[batch_size, embedding, H, W]
        fused_feature = self.conv(x)
        fused_feature = self.relu(fused_feature)

        return fused_feature   
    
class FuseViewFeatures3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FuseViewFeatures3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        # self.relu = nn.ReLU(inplace=True)

    
    
    def forward(self, x):
        # 输入x的形状为[batch_size, num_views, embedding, H, W]
        # 使用卷积层将四个视图的特征融合为[batch_size, embedding, H, W]
        # print(x.size())
        fused_feature = self.conv(x)
        # fused_feature = fused_feature+self.relu(fused_feature)
        
        return torch.squeeze(fused_feature, dim=1)
class MVCINN(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,using_decoder=False,decoder_embedding=0,num_layers_decoder=0,feature_cat=False,ablation_fusion=False,
                 concat=False,fuse_view=False,is_weighted_joint=False,SK_net=False,M=2, r=16, L=32,num_of_groups=-1):
        print('new-model! ')
        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0
        self.num_view = 4
        self.concat =concat
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.feature_cat=feature_cat
        self.is_weighted_joint=is_weighted_joint
        # Classifier head
        # self.trans_norm = nn.LayerNorm(embed_dim)
        self.weight_vector = nn.Parameter(torch.randn(self.num_view))
        self.using_decoder=using_decoder
        self.backbone=Nfnet(n_classes=5, model_path="/data/cyn/code/EyePACS/pretrain_weights/dm_nfnet_f5-ecb20ab1.pth",pretrained=True,weight_type='pth',embed_dim=None)
        # self.apply(self._init_weights)
        # load_checkpoint(model=self.backbone,checkpoint_path="/home/chenyangneng/RETFound_MAE/RETFound_cfp_weights.pth")
        
        if using_decoder:
            if ablation_fusion:
                self.fusion=MergeView_albation(in_features=3072,num_views=4)
            elif fuse_view:
                self.fusion=FuseView(in_channels=3072,out_channels=3072)
            elif not concat:    
                # self.fusion=MergeView(in_features=3072,num_views=4)
                self.fusion=FuseViewFeatures(in_channels=4*3072,out_channels=3072,SK_net=SK_net,M=M,r=r,L=L)
                # self.fusion=FuseViewFeatures3D(in_channels=4,out_channels=1)
                # self.fusion=jointLayer(in_channels=3072)
            self.backbone.model.head=nn.Identity()
            self.head=MLDecoder(num_classes=num_classes,decoder_embedding=decoder_embedding,initial_num_features=3072,num_layers_decoder=num_layers_decoder,num_of_groups=num_of_groups)
        else:
            self.head = Mlp(in_features=3072, hidden_features=4*3072,out_features=5)
        # self.jointLayer= jointLayer(stage_3_channel)
        # print(jointLayer)
        # trunc_normal_(self.cls_token, std=.02)

    def weighted_joint(self,x):
        # x= rearrange(x,'(b v) c h w -> b v c h w',v=4)
        b ,v ,c ,h ,w=x.shape
        arr = x.clone()
        arr = arr.view(b,c,2*h,2*w)
        
        arr[:,:,:h,:w] = x[:,0,:,:,:]*self.weight_vector[0]
        arr[:,:,:h,w:2*w] = x[:,1,:,:,:]*self.weight_vector[1]
        arr[:,:,h:2*h,:w] = x[:,2,:,:,:]*self.weight_vector[2]
        arr[:,:,h:2*h,w:2*w] = x[:,3,:,:,:]*self.weight_vector[3]
        return arr
    
        
    def joint(self,x):
        # x= rearrange(x,'(b v) c h w -> b v c h w',v=4)
        b ,v ,c ,h ,w=x.shape
        arr = x.clone()
        arr = arr.view(b,c,2*h,2*w)
        arr[:,:,:h,:w] = x[:,0,:,:,:]
        arr[:,:,:h,w:2*w] = x[:,1,:,:,:]
        arr[:,:,h:2*h,:w] = x[:,2,:,:,:]
        arr[:,:,h:2*h,w:2*w] = x[:,3,:,:,:]
        return arr

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def joint(self,x):
        # x= rearrange(x,'(b v) c h w -> b v c h w',v=4)
        b ,v ,c ,h ,w=x.shape
        arr = x
        arr = arr.view(b,c,2*h,2*w)
        arr[:,:,:h,:w] = x[:,0,:,:,:]
        arr[:,:,:h,w:2*w] = x[:,1,:,:,:]
        arr[:,:,h:2*h,:w] = x[:,2,:,:,:]
        arr[:,:,h:2*h,w:2*w] = x[:,3,:,:,:]
        return arr


    def _add(self,x):
        #
        x= rearrange(x,'(b v) c e -> b c (v e)',v=4)
        # x = rearrange(x, '(b v) c e -> b c v e', v=4)
        # x = torch.einsum('bcve->bce',x)

        return x
    def forward(self, x):

        
        mv_x_t = []
        
        for i in range(0,self.num_view):
            sv_x = x[:, i, :, :, :]
            sv_x_t =self.backbone(sv_x)
            mv_x_t.append(sv_x_t)
            
        if self.feature_cat:
            x_t = torch.cat(mv_x_t, dim=1)
        else:
            x_t = torch.stack(mv_x_t,1)
            if self.using_decoder:
                if self.concat:
                    x_t=self.joint(x_t)
                else:
                    x_t=self.fusion(x_t)

            else:
                x_t = torch.sum(x_t * self.weight_vector.unsqueeze(0).unsqueeze(-1), dim=1)
        
        # print(x_t.size())
        tran_cls = self.head(x_t)
        # tran_cls = self.head(concatenated_features)
        return [tran_cls],None

class jointLayer(nn.Module):
    def __init__(self,in_channels=1536):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels*2,kernel_size=3,stride=1,padding=1 , bias=True)
        self.conv2 = nn.Conv2d(in_channels*2,in_channels,kernel_size=1,stride=1,padding="same" ,bias=True)
        self.NONLocalBlock2D = NONLocalBlock2D(in_channels = in_channels,sub_sample=True)

    def joint(self,x):
        # x= rearrange(x,'(b v) c h w -> b v c h w',v=4)
        b ,v ,c ,h ,w=x.shape
        arr = x.clone()
        arr = arr.view(b,c,2*h,2*w)
        arr[:,:,:h,:w] = x[:,0,:,:,:]
        arr[:,:,:h,w:2*w] = x[:,1,:,:,:]
        arr[:,:,h:2*h,:w] = x[:,2,:,:,:]
        arr[:,:,h:2*h,w:2*w] = x[:,3,:,:,:]
        return arr
    
    def forward(self,x):
        #1.concat
        x = self.joint(x)
        #2.add
        # x = rearrange(x, '(b v) c h w -> b v c h w', v=4)
        # x = torch.einsum('bvchw->bchw',x)

        x = self.NONLocalBlock2D(x)

        return x
    
    
# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))


def DS_Combin_two(alpha1, alpha2):
        n_classes=5
        # Calculate the merger of two DS evidences
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = n_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, n_classes, 1), b[1].view(-1, 1, n_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = n_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a


class ETMC(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        self.num_view = 4
        self.num_classes=5
        self.embedding_dim=3072
        self.encoders=nn.ModuleList()
        if pretrained:
            for i in range(self.num_view):
                self.encoders.append(Nfnet(n_classes=5, model_path="/data/cyn/code/EyePACS/pretrain_weights/dm_nfnet_f5-ecb20ab1.pth",pretrained=True,weight_type='pth',embed_dim=None))
        else:
            for i in range(self.num_view):
                self.encoders.append(Nfnet(n_classes=5,pretrained=False,weight_type='pth',embed_dim=None))
        self.clf = nn.ModuleList()
        for i in range(self.num_view):
            self.clf.append(nn.Linear(self.embedding_dim, self.num_classes))
        self.clf_pseudo=nn.Linear(self.embedding_dim*self.num_view, self.num_classes)

    
    
    def forward(self, x):
        mult_view_outputs=[]
        for i in range(self.num_view):
            sv_x = x[:, i, :, :, :]
            sv_x_t =self.encoders[i](sv_x)
            mult_view_outputs.append(sv_x_t)
        outputs=[]
        for i in range(self.num_view):
            outputs.append(self.clf[i](mult_view_outputs[i]))
        pseudo_out = torch.cat(mult_view_outputs, -1)
        pseudo_out = self.clf_pseudo(pseudo_out)
        outputs.append(pseudo_out)
        alpha=[]
        for i in range(self.num_view+1):
            alpha.append(F.softplus(outputs[i])+1)
        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        
        # depth_evidence, rgb_evidence, pseudo_evidence = F.softplus(depth_out), F.softplus(rgb_out), F.softplus(pseudo_out)
        # depth_alpha, rgb_alpha, pseudo_alpha = depth_evidence+1, rgb_evidence+1, pseudo_evidence+1
        return alpha,alpha_a