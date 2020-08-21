#!/bin/bash
#Submit to GPU


directory=~/projects


#1. Original Unet denoising 
MODEL=UnetDN5
SIGMA=70
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}"
echo $CHECKPOINT
for data_test in DenoiseSet68 ; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --save ${CHECKPOINT} --model $MODEL --n_feats 32 --n_colors 1 \
--noise_sigma $SIGMA  --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/unet/unet_sigma70.pt"
done


#2. Unet clustering
TEMPLATE=UNETDN5_CLUSTER
SIGMA=70
CHECKPOINT="Test/${TEMPLATE}_Sigma${SIGMA}"
echo $CHECKPOINT
for data_test in DenoiseSet68 ; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --save ${CHECKPOINT} --model ClusterNet --template $TEMPLATE --n_feats 32 --n_colors 1 \
--noise_sigma $SIGMA  --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 --pretrain_cluster "${directory}/logs/dhp_restoration/unet/unet_sigma70.pt" \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/unet/unet_cluster_sigma70.pt"
done


#3. Unet group, group_size = 16
MODEL=UnetDN5_GROUP
SIGMA=70
S_GROUP=16
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}_Gs${S_GROUP}"
echo $CHECKPOINT
for data_test in DenoiseSet68 ; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --save ${CHECKPOINT} --model $MODEL --n_feats 32 --n_colors 1 --group_size ${S_GROUP} \
--noise_sigma $SIGMA  --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/unet/unet_group_Gs16_sigma70.pt"
done


#4. Unet factor, sic_layer = 2
MODEL=UnetDN5_FACTOR
SIGMA=70
FACTOR_LAYER=2
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}_SIC${FACTOR_LAYER}"
echo $CHECKPOINT
for data_test in DenoiseSet68 ; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --save ${CHECKPOINT} --model $MODEL --n_feats 32 --n_colors 1 --sic_layer ${FACTOR_LAYER} \
--noise_sigma $SIGMA  --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/unet/unet_factor_sic2_sigma70.pt"
done


#5. Unet factor, sic_layer = 3
MODEL=UnetDN5_FACTOR
SIGMA=70
FACTOR_LAYER=3
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}_SIC${FACTOR_LAYER}"
echo $CHECKPOINT
for data_test in DenoiseSet68 ; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --save ${CHECKPOINT} --model $MODEL --n_feats 32 --n_colors 1 --sic_layer ${FACTOR_LAYER} \
--noise_sigma $SIGMA  --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/unet/unet_factor_sic3_sigma70.pt"
done


#6. Unet DHP, compression ratio = 0.2
MODEL=DHP_UnetDN5
SIGMA=70
RATIO=0.2
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}_Ratio${RATIO}"
echo $CHECKPOINT
for data_test in DenoiseSet68 ; do
CUDA_VISIBLE_DEVICES=1 python ../main_dhp.py --save $CHECKPOINT --model $MODEL --n_feats 32 --n_colors 1 \
--noise_sigma ${SIGMA} --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/unet/unet_dhp_ratio2_sigma70.pt"
done


#7. Unet DHP, compression ratio = 0.4
MODEL=DHP_UnetDN5
SIGMA=70
RATIO=0.4
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}_Ratio${RATIO}"
echo $CHECKPOINT
for data_test in DenoiseSet68 ; do
CUDA_VISIBLE_DEVICES=1 python ../main_dhp.py --save $CHECKPOINT --model $MODEL --n_feats 32 --n_colors 1 \
--noise_sigma ${SIGMA} --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/unet/unet_dhp_ratio4_sigma70.pt"
done


#8. Unet DHP, compression ratio = 0.6
MODEL=DHP_UnetDN5
SIGMA=70
RATIO=0.6
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}_Ratio${RATIO}"
echo $CHECKPOINT
for data_test in DenoiseSet68 ; do
CUDA_VISIBLE_DEVICES=1 python ../main_dhp.py --save $CHECKPOINT --model $MODEL --n_feats 32 --n_colors 1 \
--noise_sigma ${SIGMA} --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/unet/unet_dhp_ratio6_sigma70.pt"
done





