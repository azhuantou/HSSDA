#!/usr/bin/env bash

#set -x
#NGPUS=$1
#PY_ARGS=${@:2}
#
#while true
#do
#    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#    if [ "${status}" != "0" ]; then
#        break;
#    fi
#done
#echo $PORT
#
#python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}

# export CUDA_VISIBLE_DEVICES=2,3
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# NGPUS=2
# CFG_DIR=cfgs

#CFG_NAME=kitti_models/pv_rcnn_ssl
#CFG_NAME=waymo_models/pv_rcnn_ssl

#python -m torch.distributed.launch --master_port 6666 --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --epochs 80 --batch_size 96 --workers 12 --pretrained_model /data/cqupt/code/semi_idea/checkpoint_epoch_80.pth
#python -m torch.distributed.launch --master_port 6668 --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --epochs 10 --batch_size 48 --workers 12 --pretrained_model /data/cqupt/code/openpc052/output/waymo_models/semi_supervised_exp/001_squence/voxel_rcnn_with_centerhead_dyn_voxel_001_sequence/default/ckpt/checkpoint_epoch_30.pth

#CFG_NAME=kitti_models/pv_rcnn_ssl
#python -m torch.distributed.launch --master_port 6666 --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --epochs 80 --batch_size 50 --workers 12 --pretrained_model /data/cqupt/code/openpc052/output/kitti_models/0_semi_3dioumatch_exp/scene_002/pv_rcnn_3/default/ckpt/checkpoint_epoch_80.pth

#CFG_NAME=kitti_models/voxel_rcnn_3classes_ssl
#python -m torch.distributed.launch --master_port 6666 --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --epochs 80 --batch_size 144 --workers 12 --pretrained_model /data/cqupt/code/openpc052/output/kitti_models/0_semi_3dioumatch_exp/scene_002/voxel_rcnn_3classes_1/default/ckpt/checkpoint_epoch_80.pth

#CFG_NAME=kitti_models/pv_rcnn
#python -m torch.distributed.launch --master_port 6666 --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --epochs 80 --batch_size 60 --workers 12 --extra_tag pv_rcnn_001_3

# CFG_NAME=kitti_models/pv_rcnn_ssl
# python -m torch.distributed.launch --master_port 6676 --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --epochs 80 --batch_size 60 --workers 12 --extra_tag pv_rcnn_001_3_semi_real_mid --pretrained_model /data/cqupt/code/semi_idea/output/kitti_models/pv_rcnn/pv_rcnn_001_3/ckpt/checkpoint_epoch_80.pth --labeled_frame_idx /data/cqupt/code/semi_idea/data/kitti/semi_supervised_data_3dioumatch/scene_001/3/label_idx.txt

# CFG_NAME=kitti_models/pv_rcnn_ssl
# python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --epochs 80 --batch_size 12 --workers 8 --extra_tag pv_rcnn_002_1 --pretrained_model ../output/kitti_models/pv_rcnn/pv_rcnn_002_1/ckpt/checkpoint_epoch_10.pth --labeled_frame_idx ../data/kitti/semi_supervised_data_3dioumatch/scene_0.02/1/label_idx.txt

set -x
NGPUS=2
PY_ARGS=$*  # ${@:2}

export CUDA_VISIBLE_DEVICES=2,3

python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}
