import os
import pickle
import numpy as np
import copy
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import random
import sys


def get_kitti_semi_data_based_on_3dioumatch_imageset(ratio_of_scene, split_idx):

    assert str(ratio_of_scene) in ['0.01', '0.02']
    assert split_idx in [1, 2, 3]

    root_path = Path('..')

    ori_root = root_path / 'data' / 'kitti' / 'origin_label' / 'kitti_infos_train.pkl'
    sample_idx_txt = root_path / 'data' / 'kitti' / 'ImageSets_3dioumatch' / ('train_' + str(ratio_of_scene) + '_' + str(split_idx) + '.txt')

    out_pkl_path_label = root_path / 'data' / 'kitti' / 'semi_supervised_data_3dioumatch' / ('scene_' + str(ratio_of_scene)) / str(split_idx) / 'kitti_infos_train.pkl'
    out_pkl_path_label_include_unlabel = root_path / 'data' / 'kitti' / 'semi_supervised_data_3dioumatch' / ('scene_' + str(ratio_of_scene)) / str(split_idx) / 'kitti_infos_train_include_unlabel.pkl'
    out_txt_path = root_path / 'data' / 'kitti' / 'semi_supervised_data_3dioumatch' / ('scene_' + str(ratio_of_scene)) / str(split_idx) / 'label_idx.txt'

    with open(ori_root, 'rb') as f:
        origin_train_anns = pickle.load(f)
    f.close()
    sample_count = [0, 0, 0]
    sample_info = []
    sample_info_include_unlabed_scene = []
    sample_idx_data = np.loadtxt(sample_idx_txt)[:, 0]

    sample_idx_list = []
    for i in tqdm(range(len(origin_train_anns))):
        if int(origin_train_anns[i]['point_cloud']['lidar_idx']) in sample_idx_data:
            name = origin_train_anns[i]['annos']['name']
            cur_scene_idx = origin_train_anns[i]['point_cloud']['lidar_idx']
            sample_idx_list.append(int(cur_scene_idx))
            sample_count[0] += sum(name == 'Car')
            sample_count[1] += sum(name == 'Pedestrian')
            sample_count[2] += sum(name == 'Cyclist')
            sample_info.append(copy.deepcopy(origin_train_anns[i]))

        else:
            cur_ori_anns = origin_train_anns[i]['annos']
            for key in cur_ori_anns.keys():
                flag = [False] * len(cur_ori_anns[key])
                cur_ori_anns[key] = cur_ori_anns[key][flag]

        sample_info_include_unlabed_scene.append(copy.deepcopy(origin_train_anns[i]))

    # only labeled scene info

    directory = os.path.dirname(out_pkl_path_label)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(out_pkl_path_label, 'wb') as f:
        pickle.dump(sample_info, f)
    f.close()

    np.savetxt(out_txt_path, np.array(sample_idx_list, dtype=int))

    # labeled scenes + unlabeled scenes
    with open(out_pkl_path_label_include_unlabel, 'wb') as f:
        pickle.dump(sample_info_include_unlabed_scene, f)
    f.close()

    print(sample_count)


if __name__ == '__main__':
    ratio = float(sys.argv[1])
    num = int(sys.argv[2])
    get_kitti_semi_data_based_on_3dioumatch_imageset(ratio, num)
