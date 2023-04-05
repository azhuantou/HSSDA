import os
import copy
import yaml
import torch
import numpy as np
from pathlib import Path
from easydict import EasyDict
from collections import defaultdict, OrderedDict

from .voxel_rcnn import VoxelRCNN
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils, calibration_kitti
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...datasets.augmentor.data_augmentor import DataAugmentor
from ...datasets.processor.data_processor import DataProcessor
from ...datasets.processor.point_feature_encoder import PointFeatureEncoder

from .detector3d_template import Detector3DTemplate


class VoxelRCNNSSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)

        self.label_idx = np.loadtxt(model_cfg['LABELED_FRAME_IDX'])
        self.voxel_rcnn = VoxelRCNN(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.voxel_rcnn_ema = VoxelRCNN(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        for param in self.voxel_rcnn_ema.parameters():
            param.detach_()

        self.add_module('voxel_rcnn', self.voxel_rcnn)
        self.add_module('voxel_rcnn_ema', self.voxel_rcnn_ema)

        self.get_pseudo_label = GetPseudoLabel()
        self.data_processor = TrainDataProcessor(root_path=model_cfg['ROOT_PATH'])
        self.my_global_step = 0

    def forward(self, batch_dict):
        if 'pseudo_label_flag' in batch_dict.keys():
            batch_dict['ema_training_flag'] = False
            with torch.no_grad():
                self.voxel_rcnn_ema.eval()
                for cur_module in self.voxel_rcnn_ema.module_list:
                    batch_dict = cur_module(batch_dict)
                pred_dicts, recall_dicts = self.voxel_rcnn_ema.post_processing(batch_dict)
            return pred_dicts, recall_dicts

        elif self.training:
            batch_size = batch_dict['batch_size']
            dual_threshold = batch_dict['dual_threshold'] 
            batch_dict.pop('dual_threshold')
            batch_frame_id = batch_dict['frame_id']
            labeled_idx_mask = [True if int(frame) in self.label_idx else False for frame in batch_frame_id]

            # self.voxel_rcnn_ema.eval()
            batch_dict_train = copy.deepcopy(batch_dict)
            batch_dict_aug = copy.deepcopy(batch_dict)

            with torch.no_grad():
                self.voxel_rcnn_ema.eval()
                batch_dict['ema_training_flag'] = False
                for cur_module in self.voxel_rcnn_ema.module_list:
                    batch_dict = cur_module(batch_dict)

                pred_dicts, _ = self.voxel_rcnn_ema.post_processing(batch_dict, no_recall_dict=True)
                pred_dicts_no_nms, _ = self.voxel_rcnn_ema.post_processing(
                    batch_dict, no_nms=True, no_recall_dict=True, score_thersh=0.1)

                if dual_threshold is not None:
                    for key in batch_dict_aug.keys():
                        if 'aug_' in key:
                            temp_key = key
                            key = key.replace('aug_', '')
                            batch_dict_aug[key] = batch_dict_aug[temp_key]

                    batch_dict_aug['ema_training_flag'] = False
                    for cur_module in self.voxel_rcnn_ema.module_list:
                        batch_dict_aug = cur_module(batch_dict_aug)
                    pred_dicts_aug, _ = self.voxel_rcnn_ema.post_processing(batch_dict_aug, no_recall_dict=True)
                    pseudo_label_dict = self.get_pseudo_label.filter_middle_pseudo_label(pred_dicts, pred_dicts_aug,
                                        batch_dict, batch_dict_aug['rot'], batch_dict_aug['sca'],
                                        dual_threshold, labeled_idx_mask)
                    batch_dict_train['gt_boxes'] = pseudo_label_dict['new_gt_box']
                confident_points = self.data_processor.get_removal_points_scene_no_label(pred_dicts_no_nms, batch_dict_train, labeled_idx_mask)

            batch_dict_train['points'] = confident_points
            batch_dict_train = self.data_processor.generate_datadict(batch_dict_train)

            split_points, shuffle_info = self.split_points(batch_dict_train['points'], labeled_idx_mask)
            batch_dict_train['points'] = split_points
            batch_dict_train = self.data_processor.generate_datadict(batch_dict_train, no_augmentator=True)

            if dual_threshold is not None:
                batch_dict_train['select_low_idx'] = pseudo_label_dict['select_mid_idx']
                batch_dict_train['select_low_score'] = pseudo_label_dict['select_mid_weight']
            else:
                batch_dict_train['select_low_idx'] = [[] for _ in range(batch_size)]
                batch_dict_train['select_low_score'] = [np.array([]) for _ in range(batch_size)]

            self.voxel_rcnn.train()
            for cur_module in self.voxel_rcnn.module_list:
                batch_dict_train = cur_module(batch_dict_train)

                if 'VoxelBackBone8x' == cur_module.model_cfg['NAME']:
                    self.unshuffle_3d_feature(batch_dict_train['multi_scale_3d_features'], shuffle_info)
                    self.unshuffle_sp_feature(batch_dict_train['encoded_spconv_tensor'], shuffle_info)

            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            for cur_module in self.voxel_rcnn.module_list:
                batch_dict = cur_module(batch_dict)
            pred_dicts, recall_dicts = self.voxel_rcnn.post_processing(batch_dict)
            return pred_dicts, recall_dicts
    @staticmethod
    def unshuffle_sp_feature(feature_3d, shuffle_info):
        batch_size = len(shuffle_info)
        # out_feature = copy.deepcopy(feature_3d)
        cur_feature = feature_3d
        cur_indices = cur_feature.indices

        w, h = cur_feature.spatial_shape[1], cur_feature.spatial_shape[2]
        xy_range = [0, w, 0, h]
        xy_range = torch.tensor(xy_range).to(cur_indices.device)
        x_lens = xy_range[3] - xy_range[2]
        y_lens = xy_range[1] - xy_range[0]
        center_xy = [xy_range[2] + x_lens / 2, xy_range[0] + y_lens / 2]

        area_0_center = [center_xy[0] + x_lens / 4, center_xy[1] - y_lens / 4]
        area_1_center = [center_xy[0] + x_lens / 4, center_xy[1] + y_lens / 4]
        area_2_center = [center_xy[0] - x_lens / 4, center_xy[1] + y_lens / 4]
        area_3_center = [center_xy[0] - x_lens / 4, center_xy[1] - y_lens / 4]

        area_center_list = [area_0_center, area_1_center, area_2_center, area_3_center]

        area_0_limit = [center_xy[0] + x_lens / 2, center_xy[1] - y_lens / 2]
        area_1_limit = [center_xy[0] + x_lens / 2, center_xy[1] + y_lens / 2]
        area_2_limit = [center_xy[0] - x_lens / 2, center_xy[1] + y_lens / 2]
        area_3_limit = [center_xy[0] - x_lens / 2, center_xy[1] - y_lens / 2]
        area_limit_list = [area_0_limit, area_1_limit, area_2_limit, area_3_limit]

        out_cur_indices = cur_indices.new_zeros(cur_indices.size())  # output indices after unshuffle
        for bs_id in range(batch_size):
            bs_indices_mask = cur_indices[:, 0] == bs_id
            bs_indices = cur_indices[bs_indices_mask]
            bs_shuffle_info = shuffle_info[bs_id]
            out_bs_indices = bs_indices.new_zeros(bs_indices.size())

            for ori_area, cur_area in enumerate(bs_shuffle_info):
                # if ori_area == cur_area:
                #     continue
                cur_area_limit = area_limit_list[cur_area]
                x_limit = [min(center_xy[0], cur_area_limit[0]), max(center_xy[0], cur_area_limit[0])]
                y_limit = [min(center_xy[1], cur_area_limit[1]), max(center_xy[1], cur_area_limit[1])]
                area_mask = (x_limit[0] <= bs_indices[:, 3]) & (bs_indices[:, 3] < x_limit[1]) &\
                            (y_limit[0] <= bs_indices[:, 2]) & (bs_indices[:, 2] < y_limit[1])
                bs_indices_cur_area = bs_indices[area_mask]

                target_area_center = area_center_list[ori_area]
                cur_area_center = area_center_list[cur_area]
                target_move_pace = [a - b for a, b in zip(target_area_center, cur_area_center)]
                target_move_pace = torch.tensor(target_move_pace[::-1]).to(bs_indices_cur_area.device).int()
                bs_indices_cur_area[:, 2:] = bs_indices_cur_area[:, 2:] + target_move_pace

                out_bs_indices[area_mask] = bs_indices_cur_area

            out_cur_indices[bs_indices_mask] = out_bs_indices
        feature_3d.indices = out_cur_indices
        # return out_cur_indices

    @staticmethod
    def unshuffle_3d_feature(feature_3d, shuffle_info):
        batch_size = len(shuffle_info)
        # out_feature = copy.deepcopy(feature_3d)
        for key in feature_3d.keys():
            cur_feature = feature_3d[key]
            cur_indices = cur_feature.indices

            w, h = cur_feature.spatial_shape[1], cur_feature.spatial_shape[2]
            xy_range = [0, w, 0, h]
            xy_range = torch.tensor(xy_range).to(cur_indices.device)
            x_lens = xy_range[3] - xy_range[2]
            y_lens = xy_range[1] - xy_range[0]
            center_xy = [xy_range[2] + x_lens / 2, xy_range[0] + y_lens / 2]

            area_0_center = [center_xy[0] + x_lens / 4, center_xy[1] - y_lens / 4]
            area_1_center = [center_xy[0] + x_lens / 4, center_xy[1] + y_lens / 4]
            area_2_center = [center_xy[0] - x_lens / 4, center_xy[1] + y_lens / 4]
            area_3_center = [center_xy[0] - x_lens / 4, center_xy[1] - y_lens / 4]

            area_center_list = [area_0_center, area_1_center, area_2_center, area_3_center]

            area_0_limit = [center_xy[0] + x_lens / 2, center_xy[1] - y_lens / 2]
            area_1_limit = [center_xy[0] + x_lens / 2, center_xy[1] + y_lens / 2]
            area_2_limit = [center_xy[0] - x_lens / 2, center_xy[1] + y_lens / 2]
            area_3_limit = [center_xy[0] - x_lens / 2, center_xy[1] - y_lens / 2]
            area_limit_list = [area_0_limit, area_1_limit, area_2_limit, area_3_limit]

            out_cur_indices = cur_indices.new_zeros(cur_indices.size())  # output indices after unshuffle
            for bs_id in range(batch_size):
                bs_indices_mask = cur_indices[:, 0] == bs_id
                bs_indices = cur_indices[bs_indices_mask]
                bs_shuffle_info = shuffle_info[bs_id]
                out_bs_indices = bs_indices.new_zeros(bs_indices.size())

                for ori_area, cur_area in enumerate(bs_shuffle_info):
                    # if ori_area == cur_area:
                    #     continue
                    cur_area_limit = area_limit_list[cur_area]
                    x_limit = [min(center_xy[0], cur_area_limit[0]), max(center_xy[0], cur_area_limit[0])]
                    y_limit = [min(center_xy[1], cur_area_limit[1]), max(center_xy[1], cur_area_limit[1])]
                    area_mask = (x_limit[0] <= bs_indices[:, 3]) & (bs_indices[:, 3] < x_limit[1]) &\
                                (y_limit[0] <= bs_indices[:, 2]) & (bs_indices[:, 2] < y_limit[1])
                    bs_indices_cur_area = bs_indices[area_mask]

                    target_area_center = area_center_list[ori_area]
                    cur_area_center = area_center_list[cur_area]
                    target_move_pace = [a - b for a, b in zip(target_area_center, cur_area_center)]
                    target_move_pace = torch.tensor(target_move_pace[::-1]).to(bs_indices_cur_area.device).int()
                    bs_indices_cur_area[:, 2:] = bs_indices_cur_area[:, 2:] + target_move_pace

                    out_bs_indices[area_mask] = bs_indices_cur_area

                out_cur_indices[bs_indices_mask] = out_bs_indices

            feature_3d[key].indices = out_cur_indices

    @staticmethod
    def split_points(points, label_mask):
        final_points = []
        shuffle_info = []
        points = points.cpu().numpy()
        batch_size = int(points[-1, 0]) + 1
        xy_range = np.array([-40, 40, 0, 70.4])
        x_lens = xy_range[3] - xy_range[2]
        y_lens = xy_range[1] - xy_range[0]
        center_xy = [xy_range[2] + x_lens / 2, xy_range[0] + y_lens / 2]

        area_0_center = [center_xy[0] + x_lens / 4, center_xy[1] - y_lens / 4]
        area_1_center = [center_xy[0] + x_lens / 4, center_xy[1] + y_lens / 4]
        area_2_center = [center_xy[0] - x_lens / 4, center_xy[1] + y_lens / 4]
        area_3_center = [center_xy[0] - x_lens / 4, center_xy[1] - y_lens / 4]

        area_center_list = [area_0_center, area_1_center, area_2_center, area_3_center]

        area_0_limit = [center_xy[0] + x_lens / 2, center_xy[1] - y_lens / 2]
        area_1_limit = [center_xy[0] + x_lens / 2, center_xy[1] + y_lens / 2]
        area_2_limit = [center_xy[0] - x_lens / 2, center_xy[1] + y_lens / 2]
        area_3_limit = [center_xy[0] - x_lens / 2, center_xy[1] - y_lens / 2]
        area_limit_list = [area_0_limit, area_1_limit, area_2_limit, area_3_limit]

        for bs in range(batch_size):
            bs_flag = points[:, 0] == bs
            cur_points = points[bs_flag]
            res_points = []
            shuffle_area = [0, 1, 2, 3]
            if not label_mask[bs]:
                np.random.shuffle(shuffle_area)
            for i in range(len(shuffle_area)):
                cur_area_limit = area_limit_list[i]
                x_limit = [min(center_xy[0], cur_area_limit[0]), max(center_xy[0], cur_area_limit[0])]
                y_limit = [min(center_xy[1], cur_area_limit[1]), max(center_xy[1], cur_area_limit[1])]
                cur_area_points = cur_points[(x_limit[0] <= cur_points[:, 1]) & (cur_points[:, 1] < x_limit[1])
                                         & (y_limit[0] <= cur_points[:, 2]) & (cur_points[:, 2] < y_limit[1])]

                target_area = shuffle_area[i]
                target_center = area_center_list[target_area]
                target_move_pace = [a - b for a, b in zip(target_center, area_center_list[i])]
                cur_area_points[:, 1:-2] = cur_area_points[:, 1:-2] + target_move_pace
                res_points.append(cur_area_points)
            shuffle_info.append(shuffle_area)
            res_points = np.concatenate(res_points)
            final_points.append(res_points)
        final_points = np.concatenate(final_points)
        return final_points, shuffle_info

    def get_training_loss(self):
        disp_dict = {}
        loss = 0

        loss_rpn, tb_dict = self.voxel_rcnn.dense_head.get_loss()
        loss_rcnn, tb_dict = self.voxel_rcnn.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    # def update_global_step(self):
    #     self.global_step += 1
    #     alpha = 0.999
    #     alpha = min(1 - 1 / (self.global_step + 1), alpha)  # 44080  - 2160
    #     for ema_param, param in zip(self.voxel_rcnn_ema.parameters(), self.voxel_rcnn.parameters()):
    #         ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

    @torch.no_grad()
    def update_global_step(self):
        self.my_global_step += 1
        ema_keep_rate = 0.9996
        change_global_step = 2000
        if self.my_global_step < change_global_step:
            keep_rate = (ema_keep_rate - 0.5) / change_global_step * self.my_global_step + 0.5
        else:
            keep_rate = ema_keep_rate

        student_model_dict = self.voxel_rcnn.state_dict()
        new_teacher_dict = OrderedDict()
        for key, value in self.voxel_rcnn_ema.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise NotImplementedError
        self.voxel_rcnn_ema.load_state_dict(new_teacher_dict)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            # load pretrain model
            new_key = 'voxel_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

            # elif new_key in self.state_dict():
            #     if self.state_dict()[new_key].shape[1:] == model_state_disk[key].shape[:4] and \
            #             self.state_dict()[new_key].shape[0] == model_state_disk[key].shape[-1]:
            #         update_model_state[new_key] = val.permute(1, 2, 3, 4, 0)

            new_key = 'voxel_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

            # elif new_key in self.state_dict():
            #     if self.state_dict()[new_key].shape[1:] == model_state_disk[key].shape[:4] and \
            #             self.state_dict()[new_key].shape[0] == model_state_disk[key].shape[-1]:
            #         update_model_state[new_key] = val.permute(1, 2, 3, 4, 0)

            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

            # elif new_key in self.state_dict():
            #     if self.state_dict()[new_key].shape[1:] == model_state_disk[key].shape[:4] and \
            #             self.state_dict()[new_key].shape[0] == model_state_disk[key].shape[-1]:
            #         update_model_state[new_key] = val.permute(1, 2, 3, 4, 0)

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))


class GetPseudoLabel:
    def __init__(self):
        self.id_name_map = {
            1: 'Car',
            2: 'Pedestrian',
            3: 'Cyclist'
        }

    def filter_middle_pseudo_label(self, pred_dict, pred_dict_aug, batch_dict, total_rot, total_sca, dual_threshold, label_mask):
        batch_size = len(pred_dict)
        return_dict = {}
        select_mid_box = []
        select_mid_idx = []
        select_mid_weight = []
        assert len(pred_dict) == len(pred_dict_aug)
        for bs_idx in range(batch_size):
            cur_pre_dict = pred_dict[bs_idx]
            cur_pre_dict_aug = pred_dict_aug[bs_idx]

            pred_score = cur_pre_dict['pred_scores'].cpu().numpy()
            # pred_score_aug = cur_pre_dict_aug['pred_scores']

            train_box = batch_dict['gt_boxes'][bs_idx].cpu().numpy()[:, :-1]
            train_box = train_box[np.sum(train_box, 1) != 0]
            test_box = cur_pre_dict['pred_boxes'].cpu().numpy()
            aug_test_box = cur_pre_dict_aug['pred_boxes'].cpu().numpy()

            # iou_flag to filter out pseudo-labels that overlap with gt
            if len(train_box) != 0:
                iou = iou3d_nms_utils.boxes_bev_iou_cpu(train_box, test_box)
                iou_value = np.sum(iou, axis=0)

                iou_flag = iou_value == 0
            else:
                iou_flag = np.array([True] * len(test_box))

            rot = total_rot[bs_idx].item()
            sca = total_sca[bs_idx].item()

            # labeled scenes do not need pseudo-labels
            if label_mask[bs_idx]:
                select_mid_box.append(total_rot.new_zeros((0, 8)))
                select_mid_idx.append([])
                select_mid_weight.append(np.array([]))
                continue

            # predicted number of boxes must bigger than 0 in aug scene and original scene
            elif len(aug_test_box) != 0 and len(test_box) != 0:
                raw_test_box = copy.deepcopy(test_box)
                cur_aug_box = self.random_flip_along_x(raw_test_box)
                # cur_aug_box = random_flip_along_y(cur_aug_box)
                cur_aug_box = self.global_rotation(cur_aug_box, rot)
                cur_aug_box = self.global_scaling(cur_aug_box, sca)

                iou_aug_test = iou3d_nms_utils.boxes_bev_iou_cpu(cur_aug_box, aug_test_box)
                max_iou_with_test = np.max(iou_aug_test, 1)

            # original scene and aug scene can not pair with each other
            else:
                select_mid_box.append(total_rot.new_zeros((0, 8)))
                select_mid_idx.append([])
                select_mid_weight.append(np.array([]))
                continue

            names_id = cur_pre_dict['pred_labels'].cpu().numpy()
            names = np.array([self.id_name_map[idx] for idx in names_id])

            car_idx = names == 'Car'
            ped_idx = names == 'Pedestrian'
            cyc_idx = names == 'Cyclist'

            # confidence threshold constraint
            car_mid_flag = (dual_threshold['Car'][1] < pred_score[car_idx])
            ped_mid_flag = (dual_threshold['Pedestrian'][1] < pred_score[ped_idx])
            cyc_mid_flag = (dual_threshold['Cyclist'][1] < pred_score[cyc_idx])
            select_mid_flag = np.array([True] * len(test_box))
            select_mid_flag[car_idx] = car_mid_flag
            select_mid_flag[ped_idx] = ped_mid_flag
            select_mid_flag[cyc_idx] = cyc_mid_flag

            # roi threshold constraint
            roi_score = pred_dict[bs_idx]['roi_scores'].cpu().numpy()
            roi_score_flag = np.array([True] * len(test_box))
            roi_score_flag[car_idx] = (dual_threshold['Car'][3] < roi_score[car_idx]) 
            roi_score_flag[ped_idx] = (dual_threshold['Pedestrian'][3] < roi_score[ped_idx])
            roi_score_flag[cyc_idx] = (dual_threshold['Cyclist'][3] < roi_score[cyc_idx])  

            # iou threshold constraint
            aug_mid_flag = np.array([True] * len(test_box))
            aug_mid_flag[car_idx] = (dual_threshold['Car'][5] < max_iou_with_test[car_idx]) 
            aug_mid_flag[ped_idx] = (dual_threshold['Pedestrian'][5] < max_iou_with_test[ped_idx]) 
            aug_mid_flag[cyc_idx] = (dual_threshold['Cyclist'][5] < max_iou_with_test[cyc_idx]) 

            select_mid_flag = select_mid_flag & roi_score_flag
            select_mid_flag = select_mid_flag & iou_flag
            select_mid_flag = select_mid_flag & aug_mid_flag

            select_mid_weight.append(pred_score[select_mid_flag] * roi_score[select_mid_flag])  # get soft-weight

            start_idx = len(train_box)
            mid_idx = []
            for flag_idx in range(len(select_mid_flag)):
                if select_mid_flag[flag_idx]:
                    mid_idx.append(start_idx)
                    start_idx += 1
            select_mid_idx.append(mid_idx)

            # select_gt_flag = select_mid_flag
            select_gt_box = torch.cat((cur_pre_dict['pred_boxes'][select_mid_flag],
                                       cur_pre_dict['pred_labels'][select_mid_flag].unsqueeze(1).float()), dim=1)
            select_mid_box.append(select_gt_box)

        ori_gt_box = batch_dict['gt_boxes']
        new_gt_box_list = []
        for bs_idx in range(batch_size):
            cur_gt_box = ori_gt_box[bs_idx]
            cur_gt_box = cur_gt_box[torch.sum(cur_gt_box, 1) != 0]
            if label_mask[bs_idx]:   # labeled scene do not mine pseudo-label
                new_gt_box_list.append(cur_gt_box)
                continue

            if len(cur_gt_box) > 0:
                cur_gt_box = [bb.unsqueeze(0) for bb in cur_gt_box if bb[0] != 0]
                cur_gt_box = torch.cat(cur_gt_box) if len(cur_gt_box) > 0 else torch.zeros((0, 8)).to(ori_gt_box.device)
                cur_gt_box = cur_gt_box if len(cur_gt_box.shape) > 1 else cur_gt_box.unsqueeze(0)
                cur_pseudo_box = select_mid_box[bs_idx]
                new_gt_box = torch.cat((cur_gt_box, cur_pseudo_box), dim=0)
                new_gt_box_list.append(new_gt_box)
            else:
                cur_pseudo_box = select_mid_box[bs_idx]
                new_gt_box_list.append(cur_pseudo_box)

        return_dict['select_mid_idx'] = select_mid_idx
        return_dict['select_mid_weight'] = select_mid_weight
        return_dict['select_mid_box'] = select_mid_box
        return_dict['new_gt_box'] = new_gt_box_list
        return return_dict

    @staticmethod
    def random_flip_along_x(gt_boxes):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C)
        Returns:
        """
        enable = True  # np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable:
            gt_boxes[:, 1] = -gt_boxes[:, 1]
            gt_boxes[:, 6] = -gt_boxes[:, 6]
            # points[:, 1] = -points[:, 1]

            # if gt_boxes.shape[1] > 7:
            #     gt_boxes[:, 8] = -gt_boxes[:, 8]

        return gt_boxes

    @staticmethod
    def random_flip_along_y(gt_boxes):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C)
        Returns:
        """
        enable = True  # np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable:
            gt_boxes[:, 0] = -gt_boxes[:, 0]
            gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
            # points[:, 0] = -points[:, 0]

            # if gt_boxes.shape[1] > 7:
            #     gt_boxes[:, 7] = -gt_boxes[:, 7]

        return gt_boxes  # , points

    @staticmethod
    def global_rotation(gt_boxes, noise_rotation):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C),
            rot_range: [min, max]
        Returns:
        """
        # noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        # points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
        gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[
            0]
        gt_boxes[:, 6] += noise_rotation
        # if gt_boxes.shape[1] > 7:
        #     gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
        #         np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
        #         np.array([noise_rotation])
        #     )[0][:, 0:2]

        return gt_boxes  # , points

    @staticmethod
    def global_scaling(gt_boxes, noise_scale):
        """
        Args:
            gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
            points: (M, 3 + C),
            scale_range: [min, max]
        Returns:
        """
        # if scale_range[1] - scale_range[0] < 1e-3:
        #     return gt_boxes, points
        # noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        # points[:, :3] *= noise_scale
        gt_boxes[:, :6] *= noise_scale
        return gt_boxes  # , points


class TrainDataProcessor:
    def __init__(self, root_path):
        data_path = Path('../data/kitti')
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']

        data_cfg_root = root_path / 'tools' / 'cfgs' / 'dataset_configs' / 'kitti_dataset.yaml'
        pvrcnn_data_cfg_root = root_path / 'tools' / 'cfgs' / 'kitti_models' / 'voxel_rcnn_3classes_ssl.yaml'

        data_cfg = EasyDict(yaml.load(open(data_cfg_root), Loader=yaml.FullLoader))
        pvrcnn_data_cfg = yaml.load(open(pvrcnn_data_cfg_root), Loader=yaml.FullLoader)

        cfg = self.merge_new_config({}, pvrcnn_data_cfg)
        d_cfg = cfg['DATA_CONFIG']
        DATA_AUGMENTOR = d_cfg.DATA_AUGMENTOR
        num_point_features = DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NUM_POINT_FEATURES
        self.data_augmentor = DataAugmentor(
            data_path, DATA_AUGMENTOR, self.class_names, logger=None
        )
        DATA_PROCESSOR = d_cfg.DATA_PROCESSOR
        point_cloud_range = np.array(data_cfg['POINT_CLOUD_RANGE'], dtype=np.float32)
        self.data_processor = DataProcessor(
            DATA_PROCESSOR, point_cloud_range=point_cloud_range, training=True, num_point_features=num_point_features
        )

        self.point_feature_encoder = PointFeatureEncoder(
            data_cfg['POINT_FEATURE_ENCODING'],
            point_cloud_range=point_cloud_range
        )

        self.class_names_map = {
            '1': 'Car',
            '2': 'Pedestrian',
            '3': 'Cyclist'
        }

    def merge_new_config(self, config, new_config):
        if '_BASE_CONFIG_' in new_config:
            with open(new_config['_BASE_CONFIG_'], 'r') as f:
                try:
                    yaml_config = yaml.load(f, Loader=yaml.FullLoader)
                except:
                    yaml_config = yaml.load(f)
            config.update(EasyDict(yaml_config))

        for key, val in new_config.items():
            if not isinstance(val, dict):
                config[key] = val
                continue
            if key not in config:
                config[key] = EasyDict()
            self.merge_new_config(config[key], val)

        return config

    def generate_datadict(self, batch_dict_ssl, no_augmentator=False):
        try:
            del batch_dict_ssl['aug_points']
            del batch_dict_ssl['aug_gt_boxes']
            del batch_dict_ssl['rot']
            del batch_dict_ssl['sca']
            del batch_dict_ssl['voxels']
            del batch_dict_ssl['voxel_coords']
            del batch_dict_ssl['voxel_num_points']
            del batch_dict_ssl['aug_voxels']
            del batch_dict_ssl['aug_voxel_coords']
            del batch_dict_ssl['aug_voxel_num_points']
        except:
            pass
        batch_dict_list = []
        for index in range(batch_dict_ssl['batch_size']):
            cur_batch_dict = {}
            for key in batch_dict_ssl.keys():
                if key in ['calib', 'frame_id']:
                    cur_batch_dict[key] = batch_dict_ssl[key][index]
                    continue
                elif key in ['gt_boxes', 'road_plane', 'use_lead_xyz', 'image_shape']:
                    cur_batch_dict[key] = batch_dict_ssl[key][index].cpu().numpy()
                elif key in ['points']:
                    choose = batch_dict_ssl[key][:, 0] == index
                    cur_batch_dict[key] = batch_dict_ssl[key][choose][:, 1:].cpu().numpy() \
                        if type(batch_dict_ssl[key]) is not np.ndarray else batch_dict_ssl[key][choose][:, 1:]

            cur_batch_dict['gt_boxes'] = cur_batch_dict['gt_boxes'] if len(cur_batch_dict['gt_boxes']) > 0 else np.zeros((0, 8))
            class_idx = cur_batch_dict['gt_boxes'][:, -1]
            se = class_idx != 0
            cur_batch_dict['gt_boxes'] = cur_batch_dict['gt_boxes'][se, :-1]
            class_idx = class_idx[se]
            cur_batch_dict['gt_names'] = np.array([self.class_names_map[str(int(idx))] for idx in class_idx])

            if not no_augmentator:
                cur_batch_dict['cur_epoch'] = batch_dict_ssl['cur_epoch']
                gt_boxes_mask = np.array([n in self.class_names for n in cur_batch_dict['gt_names']], dtype=np.bool_)
                cur_batch_dict = self.data_augmentor.forward(
                    data_dict={
                        **cur_batch_dict,
                        'gt_boxes_mask': gt_boxes_mask
                    }
                )

            if cur_batch_dict.get('gt_boxes', None) is not None:
                selected = common_utils.keep_arrays_by_name(cur_batch_dict['gt_names'], self.class_names)
                cur_batch_dict['gt_boxes'] = cur_batch_dict['gt_boxes'][selected]
                cur_batch_dict['gt_names'] = cur_batch_dict['gt_names'][selected]
                gt_classes = np.array([self.class_names.index(n) + 1 for n in cur_batch_dict['gt_names']],
                                      dtype=np.int32)
                gt_boxes = np.concatenate((cur_batch_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)),
                                          axis=1)
                cur_batch_dict['gt_boxes'] = gt_boxes

                if cur_batch_dict.get('gt_boxes2d', None) is not None:
                    cur_batch_dict['gt_boxes2d'] = cur_batch_dict['gt_boxes2d'][selected]

            if cur_batch_dict.get('points', None) is not None:
                cur_batch_dict = self.point_feature_encoder.forward(cur_batch_dict)

            cur_batch_dict = self.data_processor.forward(
                data_dict=cur_batch_dict
            )

            cur_batch_dict.pop('gt_names', None)
            cur_batch_dict.pop('batch_index', None)

            batch_dict_list.append(cur_batch_dict)
            try:
                del cur_batch_dict['aug_points']
                del cur_batch_dict['aug_gt_boxes']
                del cur_batch_dict['rot']
                del cur_batch_dict['sca']
                del cur_batch_dict['aug_voxels']
                del cur_batch_dict['aug_voxel_coords']
                del cur_batch_dict['aug_voxel_num_points']
            except:
                pass
        batch_dict_ssl = self.collate_batch(batch_dict_list)
        self.load_data_to_gpu(batch_dict_ssl)
        return batch_dict_ssl

    @staticmethod
    def get_removal_points_scene_no_label(pred_dicts, batch_dict, label_mask):
        res_points = []
        batch_index = batch_dict['points'][:, 0]
        for index in range(len(batch_dict['gt_boxes'])):
            gt_boxes = batch_dict['gt_boxes'][index]

            pred_boxes = pred_dicts[index]['pred_boxes']
            gt_boxes = gt_boxes[torch.sum(gt_boxes, 1) != 0]

            if label_mask[index]:  # full label scenes do not change
                points_except_box = batch_dict['points'][batch_index == index]
            elif len(gt_boxes) > 0 and len(pred_boxes) > 0:
                pre2gt = iou3d_nms_utils.boxes_iou3d_gpu(gt_boxes[:, :-1], pred_boxes)
                iou = torch.sum(pre2gt, 0)
                selected = iou <= 0
                selected_boxes = pred_boxes[selected]

                points = batch_dict['points'][batch_index == index]

                points_mask = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points[:, 1:-1].unsqueeze(0), selected_boxes.unsqueeze(0))
                # points_except_box = box_utils.remove_points_in_boxes3d(points[:, 1:-1], selected_boxes)
                points_except_box = points[points_mask.squeeze() == -1]
            else:
                if len(pred_boxes) > 0 and len(gt_boxes) <= 0:
                    points = batch_dict['points'][batch_index == index]

                    points_mask = roiaware_pool3d_utils.points_in_boxes_gpu(
                        points[:, 1:-1].unsqueeze(0), pred_boxes.unsqueeze(0))
                    points_except_box = points[points_mask.squeeze() == -1]
                else:
                    points_except_box = batch_dict['points'][batch_index == index]

            if index == 0:
                res_points = points_except_box
            else:
                res_points = torch.cat((res_points, points_except_box), dim=0)

        return res_points

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

    @staticmethod
    def load_data_to_gpu(batch_dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id', 'metadata', 'calib']:
                continue
            elif key in ['images']:
                batch_dict[key] = image_to_tensor(val).float().cuda().contiguous()
            elif key in ['image_shape']:
                batch_dict[key] = torch.from_numpy(val).int().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()
