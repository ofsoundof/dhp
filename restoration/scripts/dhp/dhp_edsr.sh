#!/bin/bash
#Submit to GPU


MODEL=DHP_EDSR
N_BLOCK=8
N_FEATS=128
N_PATCH=192
N_BATCH=16
SCALE=4
REG=1e-1
T=5e-3
LIMIT=0.02
RATIO=0.6
CHECKPOINT="${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_P${N_PATCH}B${N_BATCH}_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}_Upsampler"
echo $CHECKPOINT

CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save $CHECKPOINT --model $MODEL --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --res_scale 1 \
--patch_size $N_PATCH --batch_size $N_BATCH --n_train 32208 --n_threads 8 --data_train DIV2KSUB --data_test Set5 \
--epochs 300 --lr_decay_step 40+200 --lr 0.0001 \
--prune_threshold ${T} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} \
--save_results --print_model --reset --input_dim 64 --prune_upsampler \
--dir_save /home/thor/projects/logs \
--dir_data /home/thor/projects/data

