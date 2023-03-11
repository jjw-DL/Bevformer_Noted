import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
import numpy as np


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.voxel_size = voxel_size # [0.2, 0.2, 8]
        self.post_center_range = post_center_range # [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        self.max_num = max_num # 300
        self.score_threshold = score_threshold # None
        self.num_classes = num_classes # 10

    def encode(self):

        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num # 300

        cls_scores = cls_scores.sigmoid() # (900, 10)
        scores, indexs = cls_scores.view(-1).topk(max_num) # 取前300个
        labels = indexs % self.num_classes # 计算lables (300,)
        bbox_index = indexs // self.num_classes # 计算bbox的索引 (300,)
        bbox_preds = bbox_preds[bbox_index] # 提取预测的bbox (300, 10)
       
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range) # (300, 9) 
        final_scores = scores # (300,)
        final_preds = labels # (300,)

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask] # (300, 9)
            scores = final_scores[mask] # (300,)

            labels = final_preds[mask] # (300,)
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1] # (1, 900, 10) 1:batch size
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1] # (1, 900, 10)
        
        batch_size = all_cls_scores.size()[0] # 1
        predictions_list = []
        # 逐帧decode
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list

