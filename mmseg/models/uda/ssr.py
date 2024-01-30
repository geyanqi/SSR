# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy
import cv2
import torch.nn as nn 

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.uda.dacs import DACS
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.utils.utils import downscale_label_ratio
from mmseg.ops import resize
from segment_anything import SamPredictor, sam_model_registry


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


class AttentionCross(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim , bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2=None, mode=None):
        B, N, C = x1.shape
        if x2 is None:
            x2 = x1
        # torch.Size([2, 256, H*W, 1])
        q = self.q(x1).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,
                                                           3).contiguous()

        if self.sr_ratio > 1:
            raise NotImplementedError()
            # x_ = x2.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            # x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            # x_ = self.norm(x_)
            # kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
            #                          C // self.num_heads).permute(
            #                              2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x2).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1] # torch.Size([2, 256, 256, 1])

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale # torch.Size([2, 256, 1024, 256])
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C) # torch.Size([2, 1024, 256])
        x = self.proj(x)
        x = self.proj_drop(x) # torch.Size([2, 1024, 256])
        if mode=='attn':
            return x, attn
        else:
            return x        

class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x
    

@UDA.register_module()
class SSR(DACS):

    def __init__(self, **cfg):
        super(SSR, self).__init__(**cfg)
        
        checkpoint=dict(vit_b='pretrained/sam_vit_b_01ec64.pth', \
                        vit_l='pretrained/sam_vit_l_0b3195.pth', \
                        vit_h='pretrained/sam_vit_h_4b8939.pth')
        sam_encoder = cfg['sam_model_registry']
        self.sam_encoder = sam_encoder
        sam = sam_model_registry[sam_encoder](checkpoint=checkpoint[sam_encoder])
        self.predictor = sam.image_encoder.to('cuda')
        del sam
        
        self.embedding_layer = {}
        self.fusion_layer = {}

        self.embedding_layer[str(0)] = MLP(256, 64)
        self.embedding_layer[str(1)] = MLP(256, 128)
        self.embedding_layer[str(2)] = MLP(256, 320)
        self.embedding_layer[str(3)] = MLP(256, 512)

        self.fusion_layer[str(0)] = AttentionCross(64, num_heads=1) 
        self.fusion_layer[str(1)] = AttentionCross(128, num_heads=2) 
        self.fusion_layer[str(2)] = AttentionCross(320, num_heads=5) 
        self.fusion_layer[str(3)] = AttentionCross(512, num_heads=8) 

        self.embedding_layer = nn.ModuleDict(self.embedding_layer)
        self.fusion_layer = nn.ModuleDict(self.fusion_layer)
        self.stop_stage = cfg.get('stop_stage', [0,1,2,3])
    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      rare_class=None,
                      valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        self.update_debug_state()
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        
        # ------------------------------------------------------------------
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        
        pseudo_label, pseudo_weight = None, None
        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().generate_pseudo_label(
            target_img, target_img_metas)
        seg_debug['Target'] = self.get_ema_model().debug_output

        pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
            ema_logits)
        del ema_logits

        pseudo_weight = self.filter_valid_pseudo_region(
            pseudo_weight, valid_pseudo_mask)
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mixed_seg_weight = pseudo_weight.clone()
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack(
                    (gt_semantic_seg[i][0], pseudo_label[i])))
            _, mixed_seg_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        del gt_pixel_weight
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        # ------------------------------------------------------------------

        # -------------Apply SAM-------------
        # src enhance 
        sam_src_img = resize(input=img,
                size=(1024,1024),
                mode='bilinear',
                align_corners=self.model.decode_head.align_corners)

        # tar enhance
        sam_tar_img = resize(input=mixed_img,
                size=(1024,1024),
                mode='bilinear',
                align_corners=self.model.decode_head.align_corners)
        
        sam_img = torch.cat((sam_src_img, sam_tar_img), dim=0)  
        with torch.no_grad():
            if self.sam_encoder == 'vit_b':
                src_edge_features, tar_edge_features = self.predictor(sam_img).chunk(2)
            else:
                src_edge_features = self.predictor(sam_src_img)
                tar_edge_features = self.predictor(sam_tar_img)
            #tar_edge_features = self.predictor(sam_tar_img)
        del sam_src_img, sam_tar_img, sam_img
        # -------------Apply SAM-------------

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        seg_debug['Source'] = self.get_model().debug_output
        clean_losses['decode.loss_seg'] = clean_losses['decode.loss_seg'] 
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)

        # src enhance 
        # ------------------------------------------------------------------
        edge_features = src_edge_features

        enhanced_feats = []
        for inj_index in range(4):
            if inj_index in self.stop_stage:
                embedding = self.embedding_layer[f'{inj_index}'](edge_features)
                feat = src_feat[inj_index]
                b, c, h, w = feat.shape
                feat = feat.flatten(2).transpose(1, 2).contiguous()
                fusion_feat = self.fusion_layer[f'{inj_index}'](feat, x2=embedding)
                fusion_feat = fusion_feat.transpose(1,2).reshape(b,c,h,w)
                enhanced_feats.append(fusion_feat + src_feat[inj_index])
            else:
                enhanced_feats.append(src_feat[inj_index])


        enhanced_src_logits = self.get_model().decode_head(enhanced_feats)
        enhanced_loss = self.get_model().decode_head.losses(enhanced_src_logits, gt_semantic_seg)
        enhanced_loss['loss_seg'] = enhanced_loss['loss_seg'] 
        enhanced_src_losses = add_prefix(enhanced_loss, 'enhanced')
        enhanced_src_loss, enhanced_src_log_vars = self._parse_losses(enhanced_src_losses)
        log_vars.update(enhanced_src_log_vars)
        # ------------------------------------------------------------------
        # src enhance  
        (enhanced_src_loss + clean_loss).backward(retain_graph=self.enable_fdist)
        del src_edge_features

        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img,
            img_metas,
            mixed_lbl,
            seg_weight=mixed_seg_weight,
            return_feat=True,
        )
        mix_losses['decode.loss_seg'] = mix_losses['decode.loss_seg'] 
        mix_feats = mix_losses.pop('features')
        seg_debug['Mix'] = self.get_model().debug_output
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        
        # ------------------------------------------------------------------
        # tar enhance!
        edge_features = tar_edge_features

        enhanced_feats = []
        for inj_index in range(4):
            if inj_index in self.stop_stage:
                embedding = self.embedding_layer[f'{inj_index}'](edge_features)
                feat = mix_feats[inj_index]
                b, c, h, w = feat.shape
                feat = feat.flatten(2).transpose(1, 2).contiguous()
                fusion_feat = self.fusion_layer[f'{inj_index}'](feat, x2=embedding)
                fusion_feat = fusion_feat.transpose(1,2).reshape(b,c,h,w)
                enhanced_feats.append(fusion_feat + mix_feats[inj_index])
            else:
                enhanced_feats.append(mix_feats[inj_index])


        enhanced_mix_logits = self.get_model().decode_head(enhanced_feats)
        enhanced_loss = self.get_model().decode_head.losses(enhanced_mix_logits, mixed_lbl, seg_weight=mixed_seg_weight)
        enhanced_loss['loss_seg'] = enhanced_loss['loss_seg'] 
        enhanced_mix_losses = add_prefix(enhanced_loss, 'enhanced_mix')
        enhanced_mix_loss, enhanced_mix_log_vars = self._parse_losses(enhanced_mix_losses)
        log_vars.update(enhanced_mix_log_vars)
        # ------------------------------------------------------------------
        
        (enhanced_mix_loss + mix_loss).backward()
        del tar_edge_features
        # Masked Training
        if self.enable_masking and self.mask_mode.startswith('separate'):
            masked_loss = self.mic(self.get_model(), img, img_metas,
                                   gt_semantic_seg, target_img,
                                   target_img_metas, valid_pseudo_mask,
                                   pseudo_label, pseudo_weight)
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, 'masked')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()

        self.local_iter += 1

        return log_vars
