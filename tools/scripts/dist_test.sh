

export CUDA_VISIBLE_DEVICES=1,2

NGPUS=2
CFG_DIR=cfgs

CFG_NAME=kitti_models/voxel_rcnn_3classes_ssl

python -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --batch_size 12 --workers 8 --eval_all --ckpt_dir /data/cqupt/code/semi_idea/output/kitti_models/voxel_rcnn_3classes_ssl/default/ckpt
#CFG_NAME=waymo_models/pv_rcnn_ssl
#python -m torch.distributed.launch --master_port 6666 --nproc_per_node=${NGPUS} test.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --batch_size 40 --workers 12 --ckpt /data/cqupt/code/semi_idea/output/waymo_models/pv_rcnn_ssl/default/ckpt/checkpoint_epoch_9.pth


