from functools import partial
import copy
import numpy as np
from skimage import transform

from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict
        
    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        aug_data_dict = copy.deepcopy(data_dict)
        rot_range = [-0.78539816, 0.78539816]
        scale_range = [0.91, 1.12]

        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        cur_gt_box = aug_data_dict['gt_boxes']
        points = aug_data_dict['points']

        cur_gt_box, points = self.random_flip_along_x(cur_gt_box, points)
        # cur_gt_box, points = random_flip_along_y(cur_gt_box, points)
        cur_gt_box, points = self.global_rotation(cur_gt_box, points, noise_rotation)
        cur_gt_box, points = self.global_scaling(cur_gt_box, points, noise_scale)

        aug_data_dict['points'] = points
        aug_data_dict['gt_boxes'] = cur_gt_box
        aug_data_dict['rot'] = noise_rotation
        aug_data_dict['sca'] = noise_scale

        for cur_processor in self.data_processor_queue:
            aug_data_dict = cur_processor(data_dict=aug_data_dict)

        data_dict['aug_points'] = aug_data_dict['points']

        data_dict['aug_gt_boxes'] = aug_data_dict['gt_boxes']
        try:
            data_dict['aug_voxels'] = aug_data_dict['voxels']
            data_dict['aug_voxel_coords'] = aug_data_dict['voxel_coords']
            data_dict['aug_voxel_num_points'] = aug_data_dict['voxel_num_points']
        except:
            pass
        data_dict['rot'] = aug_data_dict['rot']
        data_dict['sca'] = aug_data_dict['sca']

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict

    @staticmethod
    def random_flip_along_x(gt_boxes, points):
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
            points[:, 1] = -points[:, 1]

            # if gt_boxes.shape[1] > 7:
            #     gt_boxes[:, 8] = -gt_boxes[:, 8]

        return gt_boxes, points

    @staticmethod
    def random_flip_along_y(gt_boxes, points):
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
            points[:, 0] = -points[:, 0]

            # if gt_boxes.shape[1] > 7:
            #     gt_boxes[:, 7] = -gt_boxes[:, 7]

        return gt_boxes, points

    @staticmethod
    def global_rotation(gt_boxes, points, noise_rotation):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C),
            rot_range: [min, max]
        Returns:
        """
        # noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
        gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[
            0]
        gt_boxes[:, 6] += noise_rotation
        # if gt_boxes.shape[1] > 7:
        #     gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
        #         np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
        #         np.array([noise_rotation])
        #     )[0][:, 0:2]

        return gt_boxes, points

    @staticmethod
    def global_scaling(gt_boxes, points, noise_scale):
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
        points[:, :3] *= noise_scale
        gt_boxes[:, :6] *= noise_scale
        return gt_boxes, points

