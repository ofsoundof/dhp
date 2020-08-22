#!/bin/bash
#Submit to GPU


directory=~/projects


#1. Original DnCNN denoising 
MODEL=DNCNN
SIGMA=70
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}"
echo $CHECKPOINT
for data_test in DenoiseSet68; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --model $MODEL --m_blocks 15 --n_feats 64 --n_colors 1 --bn True \
--noise_sigma $SIGMA --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/dhp/restoration/model_zoo/dncnn/dncnn_sigma70.pt"
done


#2. DnCNN clustering
TEMPLATE=DNCNN_CLUSTER
SIGMA=70
export CHECKPOINT="Test/${TEMPLATE}_Sigma${SIGMA}"
echo $CHECKPOINT
for data_test in DenoiseSet68; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --template $TEMPLATE --model ClusterNet --m_blocks 15 --n_feats 64 --n_colors 1 --bn True \
--noise_sigma $SIGMA --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 --pretrain_cluster "${directory}/dhp/restoration/model_zoo/dncnn/dncnn_sigma70.pt" \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/dhp/restoration/model_zoo/dncnn/dncnn_cluster_sigma70.pt"
done


#3. DnCNN group, group_size = 16
MODEL=DNCNN_GROUP
SIGMA=70
S_GROUP=16
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}_Gs${S_GROUP}"
echo $CHECKPOINT
for data_test in DenoiseSet68; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --model $MODEL --m_blocks 15 --n_feats 64 --n_colors 1 --bn True --group_size ${S_GROUP} \
--noise_sigma $SIGMA --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/dhp/restoration/model_zoo/dncnn/dncnn_group_Gs16_sigma70.pt"
done


#4. DnCNN factor, sic_layer = 2
MODEL=DNCNN_FACTOR
SIGMA=70
FACTOR_LAYER=2
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}_SIC${FACTOR_LAYER}"
echo $CHECKPOINT
for data_test in DenoiseSet68; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --model $MODEL --m_blocks 15 --n_feats 64 --n_colors 1 --bn True --sic_layer ${FACTOR_LAYER} \
--noise_sigma $SIGMA --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/dhp/restoration/model_zoo/dncnn/dncnn_factor_sic2_sigma70.pt"
done


#5. DnCNN factor, sic_layer = 2
MODEL=DNCNN_FACTOR
SIGMA=70
FACTOR_LAYER=3
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}_SIC${FACTOR_LAYER}"
echo $CHECKPOINT
for data_test in DenoiseSet68; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --model $MODEL --m_blocks 15 --n_feats 64 --n_colors 1 --bn True --sic_layer ${FACTOR_LAYER} \
--noise_sigma $SIGMA --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/dhp/restoration/model_zoo/dncnn/dncnn_factor_sic3_sigma70.pt"
done


#6. DnCNN DHP, compression ratio = 0.2
MODEL=DHP_DNCNN
SIGMA=70
RATIO=0.2
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}_Ratio${RATIO}"
echo $CHECKPOINT
for data_test in DenoiseSet68; do
CUDA_VISIBLE_DEVICES=1 python ../main_dhp.py --save $CHECKPOINT --model $MODEL --m_blocks 15 --n_feats 64 --n_colors 1 --bn True \
--noise_sigma $SIGMA --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/dhp/restoration/model_zoo/dncnn/dncnn_dhp_ratio2_sigma70.pt"
done


#7. DnCNN DHP, compression ratio = 0.4
MODEL=DHP_DNCNN
SIGMA=70
RATIO=0.4
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}_Ratio${RATIO}"
echo $CHECKPOINT
for data_test in DenoiseSet68; do
CUDA_VISIBLE_DEVICES=1 python ../main_dhp.py --save $CHECKPOINT --model $MODEL --m_blocks 15 --n_feats 64 --n_colors 1 --bn True \
--noise_sigma $SIGMA --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/dhp/restoration/model_zoo/dncnn/dncnn_dhp_ratio4_sigma70.pt"
done


#8. DnCNN DHP, compression ratio = 0.6
MODEL=DHP_DNCNN
SIGMA=70
RATIO=0.6
CHECKPOINT="Test/${MODEL}_Sigma${SIGMA}_Ratio${RATIO}"
echo $CHECKPOINT
for data_test in DenoiseSet68; do
CUDA_VISIBLE_DEVICES=1 python ../main_dhp.py --save $CHECKPOINT --model $MODEL --m_blocks 15 --n_feats 64 --n_colors 1 --bn True \
--noise_sigma $SIGMA --scale 1 --data_test ${data_test} --ext bin \
--save_results --input_dim 128 \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/dhp/restoration/model_zoo/dncnn/dncnn_dhp_ratio6_sigma70.pt"
done



