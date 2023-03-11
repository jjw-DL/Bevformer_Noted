# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder) # BEVFormerLayer
        self.decoder = build_transformer_layer_sequence(decoder) # DetectionTransformerDecoder
        self.embed_dims = embed_dims # 256
        self.num_feature_levels = num_feature_levels # 4
        self.num_cams = num_cams # 6
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev # True
        self.use_shift = use_shift # True
        self.use_can_bus = use_can_bus # True
        self.can_bus_norm = can_bus_norm # True
        self.use_cams_embeds = use_cams_embeds # True

        self.two_stage_num_proposals = two_stage_num_proposals # 300
        self.init_layers()
        self.rotate_center = rotate_center # [100, 100]

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims)) # (4, 256)
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims)) # (6, 256)
        self.reference_points = nn.Linear(self.embed_dims, 3) # (256, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        ) # 18-->128-->256
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims)) # LN

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0) # 1
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) # (22500, 256)-->(22500, 1, 256)-->(22500, 1, 256)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # (1, 256, 150, 150)-->(1, 256, 22500)-->(22500, 1, 256)

        # obtain rotation angle and shift with ego motion
        delta_x = kwargs['img_metas'][0]['can_bus'][0] # eg:0 相对值 4.067
        delta_y = kwargs['img_metas'][0]['can_bus'][1] # eg:0 相对值 -2.171
        ego_angle = kwargs['img_metas'][0]['can_bus'][-2] / np.pi * 180 # eg:332.57 绝对值 332.16
        rotation_angle = kwargs['img_metas'][0]['can_bus'][-1] # eg:0 相对值 -0.413
        grid_length_y = grid_length[0] # 0.6826 = 102.4 / 150 表示一个网格代表的实际大小
        grid_length_x = grid_length[1] # 0.6826
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2) # 计算偏移距离 eg:4.6109
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180 # 计算转过的偏航角 -28.09
        if translation_angle < 0:
            translation_angle += 360 # 331.907
        bev_angle = ego_angle - translation_angle # BEV下的偏航角 332.16 - 331.907 = 0.2572
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h # BEV特征图上的偏移量（实际长度/单位长度/特征图大小）eg:0.045
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w # 0.0002
        shift_y = shift_y * self.use_shift # 相对于上一帧自车坐标系的偏移量
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor([shift_x, shift_y]) # eg:[0.0002, 0.0450]
        

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2) # (1, 22500, 256)-->(22500, 1, 256)
            if self.rotate_prev_bev:
                num_prev_bev = prev_bev.size(1) # 1
                # (22500, 1, 256)-->(150, 150, 256)-->(256, 150, 150)
                prev_bev = prev_bev.reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                prev_bev = rotate(prev_bev, rotation_angle,
                                  center=self.rotate_center) # 旋转特征图：(256, 150, 150)
                # (256, 150, 150)-->(150, 150, 256)-->(22500, 1, 256)
                prev_bev = prev_bev.permute(1, 2, 0).reshape(
                    bev_h * bev_w, num_prev_bev, -1)

        # add can bus signals
        can_bus = bev_queries.new_tensor(kwargs['img_metas'][0]['can_bus'])[
            None, None, :] # 获取该帧的can bus信息 (1, 1, 18)
        can_bus = self.can_bus_mlp(can_bus) # 对can bus进行编码-->(1, 1, 256)
        bev_queries = bev_queries + can_bus * self.use_can_bus # (22500, 1, 256) 为query增加位置信息

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape # (1, 6, 256, 23, 40)
            spatial_shape = (h, w) # [23, 40]
            feat = feat.flatten(3).permute(1, 0, 3, 2) # (1, 6, 256, 920)-->(6, 1, 920, 256)
            if self.use_cams_embeds:
                # (6, 1, 920, 256) + (6, 1, 1, 256) --> (6, 1, 920, 256)
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype) # 在图像特征上融入camera信息
            # (6, 1, 920, 256) + (1, 1, 1, 256) --> (6, 1, 920, 256)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype) # 增加level信息
            spatial_shapes.append(spatial_shape) # [23, 40] 添加特征图的空间shape到shape list中
            feat_flatten.append(feat) # 添加拉直后的特征图到特征list中, 只有一层

        feat_flatten = torch.cat(feat_flatten, 2) # 拼接特征图 (6, 1, 920, 256)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device) # [[23, 40]]
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1])) # [0]

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims) --> (6, 920, 1, 256)
        bev_embed = self.encoder(
            bev_queries, # (22500, 1, 256)
            feat_flatten, # (6, 920, 1, 256)
            feat_flatten, #  (6, 920, 1, 256)
            bev_h=bev_h, # 150
            bev_w=bev_w, # 150
            bev_pos=bev_pos, # (22500, 1, 256)
            spatial_shapes=spatial_shapes, # [[23, 40]]
            level_start_index=level_start_index, # [0]
            prev_bev=prev_bev, # None和[22500, 1, 256]
            shift=shift, # [0, 0]和[0.0002, 0.0450]
            **kwargs
        ) # --> (1, 22500, 256)

        return bev_embed # (1, 22500, 256)

    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_embed = self.get_bev_features(
            mlvl_feats, # List[(1, 6, 256, 23, 40)]
            bev_queries, # (22500, 256)
            bev_h, # 150
            bev_w, # 150
            grid_length=grid_length, # (0.6826, 0.6826)
            bev_pos=bev_pos, # (1, 256, 150, 150)
            prev_bev=prev_bev, # (1, 22500, 256)
            **kwargs) # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0) # 1
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1) # (900, 156)和(900, 156）
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1) # (1, 900, 156)
        query = query.unsqueeze(0).expand(bs, -1, -1) # (1, 900, 156)
        reference_points = self.reference_points(query_pos) # 256-->3 (1, 900, 3)
        reference_points = reference_points.sigmoid() # (1, 900, 3)
        init_reference_out = reference_points # 初始参考点(随机初始化)

        query = query.permute(1, 0, 2) # (900, 1, 256)
        query_pos = query_pos.permute(1, 0, 2) # (900, 1, 256)
        bev_embed = bev_embed.permute(1, 0, 2) # (22500, 1, 256)
        
        # (6, 900, 1, 256)和(6, 1, 900, 3)
        inter_states, inter_references = self.decoder(
            query=query, # (900, 1, 256)
            key=None, 
            value=bev_embed, # (22500, 1, 256)
            query_pos=query_pos, # (900, 1, 256)
            reference_points=reference_points, # (1, 900, 3)
            reg_branches=reg_branches, # 6层
            cls_branches=cls_branches, # None
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device), # [[150, 150]]
            level_start_index=torch.tensor([0], device=query.device), # [0]
            **kwargs)

        inter_references_out = inter_references # 内部decoder layer输出的参考点 (6, 1, 900, 3)

        """
        bev_embed:(22500, 1, 256) bev的拉直嵌入
        inter_states:(6, 900, 1, 256) 内部decoder layer输出的object query
        init_reference_out:(1, 900, 3) 随机初始化的参考点（可学习）
        inter_references_out:(6, 1, 900, 3) 内部decoder layer输出的参考点
        """
        return bev_embed, inter_states, init_reference_out, inter_references_out

