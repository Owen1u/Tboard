
# cpu
python train/train.py

# gpu
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=51676 train/train.py