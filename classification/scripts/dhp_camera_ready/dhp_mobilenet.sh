#!/bin/bash
#Submit to GPU

directory=~/projects
dir_data="${directory}/data/"
dir_save="${directory}/logs/"

# MobileNet
MODEL=MobileNet_DHP
BATCH=64
WIDTH=2.0
TEMPLATE=Tiny_ImageNet
REG=5e-5
T=2e-3
LIMIT=0.02
RATIO=0.24
CHECKPOINT=${MODEL}_${TEMPLATE}_B${BATCH}_W${WIDTH}_Drop_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save $CHECKPOINT --template "${TEMPLATE}" --model ${MODEL} --batch_size ${BATCH} --epochs 220 --decay step-50-70+step-200-205-210-215 --width_mult ${WIDTH} \
--prune_threshold ${T} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} --prune_classifier --linear_percentage 0.45 \
--dir_save ${dir_save} --dir_data ${dir_data}


# MobileNet, Ratio=0.10, 0.30, 0.50
MODEL=MobileNet_DHP
BATCH=64
WIDTH=1.0
TEMPLATE=Tiny_ImageNet
REG=5e-5
T=2e-3
LIMIT=0.02
RATIO=0.5
CHECKPOINT=${MODEL}_${TEMPLATE}_B${BATCH}_W${WIDTH}_Drop_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save $CHECKPOINT --template "${TEMPLATE}" --model ${MODEL} --batch_size ${BATCH} --epochs 220 --decay step-50-70+step-200-205-210-215 --width_mult ${WIDTH} \
--prune_threshold ${T} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} --prune_classifier --linear_percentage 0.4 \
--dir_save ${dir_save} --dir_data ${dir_data}








