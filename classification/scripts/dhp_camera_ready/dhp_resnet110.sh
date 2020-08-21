#!/bin/bash
#Submit to GPU

directory=~/projects
dir_data="${directory}/data/"
dir_save="${directory}/logs/"

# ResNet110, Ratio=0.5
MODEL=ResNet_DHP_SHARE
LAYER=110
BATCH=64
TEMPLATE=CIFAR10
REG=1e-4
T=5e-3
LIMIT=0.02
RATIO=0.5
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}_B${BATCH}_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save $CHECKPOINT --template "${TEMPLATE}_ResNet" --model ${MODEL} --batch_size ${BATCH} --epochs 300 --decay step-50+step-150-225 \
--depth ${LAYER} --prune_threshold ${T} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} --print_model \
--dir_save ${dir_save} --dir_data ${dir_data}







