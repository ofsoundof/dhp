#!/bin/bash
#Submit to GPU

#for T in 2e-3 5e-3; do
for RATIO in 0.2 0.4 0.6; do
<<COMMENT
MODEL=SRResNet_DHP_SHARE
N_BLOCK=16
N_FEATS=64
N_PATCH=96
N_BATCH=16
SCALE=4
REG=1e-1
T=5e-3
LIMIT=0.02
#RATIO=0.8
CHECKPOINT="${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_P${N_PATCH}B${N_BATCH}_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}_Upsampler"
echo $CHECKPOINT

qsub -N $CHECKPOINT ./vpython.sh ../main_dhp.py --save $CHECKPOINT --model $MODEL --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} \
--patch_size $N_PATCH --batch_size $N_BATCH --n_train 32208 --n_threads 8 --data_train DIV2KSUB --data_test Set5 \
--epochs 300 --lr_decay_step 40+200 --lr 0.0001 \
--prune_threshold ${T} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} \
--save_results --print_model --reset --input_dim 64 --prune_upsampler


MODEL=EDSR_DHP
N_BLOCK=8
N_FEATS=128
N_PATCH=192
N_BATCH=16
SCALE=4
REG=1e-1
T=5e-3
LIMIT=0.02
#RATIO=0.8
CHECKPOINT="${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_P${N_PATCH}B${N_BATCH}_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}_Upsampler"
echo $CHECKPOINT

qsub -N $CHECKPOINT ./vpython.sh ../main_dhp.py --save $CHECKPOINT --model $MODEL --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --res_scale 1 \
--patch_size $N_PATCH --batch_size $N_BATCH --n_train 32208 --n_threads 8 --data_train DIV2KSUB --data_test Set5 \
--epochs 300 --lr_decay_step 40+200 --lr 0.0001 \
--prune_threshold ${T} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} \
--save_results --print_model --reset --input_dim 64 --prune_upsampler
COMMENT

MODEL=SRResNet_DHP_SHARE
N_BLOCK=16
N_FEATS=64
N_PATCH=96
N_BATCH=16
SCALE=4
REG=1e-1
T=5e-3
LIMIT=0.02
#RATIO=0.8
CHECKPOINT="${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_P${N_PATCH}B${N_BATCH}_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}"
echo $CHECKPOINT

qsub -N $CHECKPOINT ./vpython.sh ../main_dhp.py --save $CHECKPOINT --model $MODEL --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} \
--patch_size $N_PATCH --batch_size $N_BATCH --n_train 32208 --n_threads 8 --data_train DIV2KSUB --data_test Set5 \
--epochs 300 --lr_decay_step 40+200 --lr 0.0001 \
--prune_threshold ${T} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} \
--save_results --print_model --reset --input_dim 64 


MODEL=EDSR_DHP
N_BLOCK=8
N_FEATS=128
N_PATCH=192
N_BATCH=16
SCALE=4
REG=1e-1
T=5e-3
LIMIT=0.02
#RATIO=0.8
CHECKPOINT="${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_P${N_PATCH}B${N_BATCH}_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}"
echo $CHECKPOINT

qsub -N $CHECKPOINT ./vpython.sh ../main_dhp.py --save $CHECKPOINT --model $MODEL --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --res_scale 1 \
--patch_size $N_PATCH --batch_size $N_BATCH --n_train 32208 --n_threads 8 --data_train DIV2KSUB --data_test Set5 \
--epochs 300 --lr_decay_step 40+200 --lr 0.0001 \
--prune_threshold ${T} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} \
--save_results --print_model --reset --input_dim 64 
done
#done





