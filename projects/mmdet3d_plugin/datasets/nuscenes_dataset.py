import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This dataset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length # 3
        self.overlap_test = overlap_test # False
        self.bev_size = bev_size # (150, 150)

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []

        # temporal aug
        prev_indexs_list = list(range(index-self.queue_length, index)) # eg:index=19436-->19433,19434,19435
        random.shuffle(prev_indexs_list) # 将索引打乱
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True) # 将第一帧之后的索引逆序排列(从大到小)

        input_dict = self.get_data_info(index) # 获取当前帧的info
        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx'] # 帧id eg: 28
        scene_token = input_dict['scene_token'] # 场景token
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict) # 进入data pipeline
        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None # 如果该帧不存在目标物体则直接返回None
        data_queue.insert(0, example) # 在数据队列中插入当前帧的info
        for i in prev_indexs_list: # [19435, 19434]
            i = max(0, i)
            input_dict = self.get_data_info(i) 
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict) # 增加预处理字段
                example = self.pipeline(input_dict) # 进入data pipeline
                if self.filter_empty_gt and \
                        (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                    return None # 如果该帧不存在目标物体则直接返回None
                frame_idx = input_dict['frame_idx'] # 获取帧id
            data_queue.insert(0, copy.deepcopy(example)) # 在数据队列中插入前一帧的info
        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'].data for each in queue] # 从队列中取出图像: List[3 (6, 3, 736, 1280)]
        metas_map = {} # 初始化metas的dict
        prev_pos = None
        prev_angle = None
        # 逐帧处理
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data # 获取该帧的img_metas
            if i == 0: # 如果是第一帧
                metas_map[i]['prev_bev'] = False # 将prev_bev设置为False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3]) # 从can bus中获取pos信息
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1]) # 从can bus中获取yaw角信息 角度单位
                metas_map[i]['can_bus'][:3] = 0 # 将can bus的pos置0
                metas_map[i]['can_bus'][-1] = 0 # 将can bus的yaw置0
            else:
                metas_map[i]['prev_bev'] = True # 将prev_bev设置为True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3]) # 从can bus中获取pos信息
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1]) # 从can bus中获取yaw角信息
                metas_map[i]['can_bus'][:3] -= prev_pos # 计算与前一帧的相对pos
                metas_map[i]['can_bus'][-1] -= prev_angle # 计算与前一帧的相对yaw
                prev_pos = copy.deepcopy(tmp_pos) # 更新前一帧的pos和angle
                prev_angle = copy.deepcopy(tmp_angle)

        # 统一到最后的curr帧上
        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True) # (3, 6, 3, 736, 1280)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True) # 3帧的img_metas以index作为key
        # 重新赋值新的queue
        queue = queue[-1] # img_meats(3帧), gt_bboxes_3d, gt_labels_3d, img(3帧)
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'], # 当前帧的token
            pts_filename=info['lidar_path'], # lidar的路径
            sweeps=info['sweeps'], # 过渡帧的info
            ego2global_translation=info['ego2global_translation'], # ego到global的平移
            ego2global_rotation=info['ego2global_rotation'], # ego到global的旋转
            prev_idx=info['prev'], # 前一帧的token
            next_idx=info['next'], # 下一帧的token
            scene_token=info['scene_token'], # 场景token
            can_bus=info['can_bus'], # can bus信息 (18,) [pos, rotation, accel, rotation_rate, vel, 0, 0]
            frame_idx=info['frame_idx'], # 帧id
            timestamp=info['timestamp'] / 1e6, # 时间戳
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T # 右乘
                lidar2cam_rt = np.eye(4)
                # 右乘推导
                lidar2cam_rt[:3, :3] = lidar2cam_r.T # P_l = P_c * (R_l_c)^(T) - t_l_c * (R_l_c)^(T)
                lidar2cam_rt[3, :3] = -lidar2cam_t
                # 相机内参
                intrinsic = cam_info['cam_intrinsic'] 
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T) # 左乘，所以要乘转置
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T) # 左乘-->lidar到camera图像平面的投影矩阵

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts, # lidar到img的投影矩阵
                    cam_intrinsic=cam_intrinsics, # 相机内参
                    lidar2cam=lidar2cam_rts, # lidar到cam的投影矩阵
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index) # 标注信息
            input_dict['ann_info'] = annos # 记录标准信息

        rotation = Quaternion(input_dict['ego2global_rotation']) # 自车到全局的旋转
        translation = input_dict['ego2global_translation'] # 自车到全局的平移
        can_bus = input_dict['can_bus'] # can bus信息 (18,) [pos, rotation, accel, rotation_rate, vel, 0, 0]
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180 # 计算yaw角
        if patch_angle < 0:
            patch_angle += 360 # 以x轴为起点，逆时针为正
        can_bus[-2] = patch_angle / 180 * np.pi # 弧度
        can_bus[-1] = patch_angle # 角度

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
