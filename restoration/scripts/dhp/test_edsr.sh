#!/bin/bash
#Submit to GPU


directory=~/projects


for test_set in Set5 Set14 B100 Urban100 DIV2K; do
# 1. EDSR Original
MODEL=EDSR
N_BLOCK=8
N_FEATS=128
SCALE=4
export CHECKPOINT="Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --res_scale 1 \
--input_dim 128 --data_test $test_set --save_results \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/edsr/edsr_scale4.pt"
#done


# 2. EDSR cluster
#for test_set in Set5; do
TEMPLATE=EDSR_CLUSTER
N_BLOCK=8
N_FEATS=128
SCALE=4
export CHECKPOINT="Test/${TEMPLATE}_X${SCALE}_L${N_BLOCK}F${N_FEATS}"
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main.py --template $TEMPLATE --model ClusterNet --save $CHECKPOINT --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --res_scale 1 \
--input_dim 128 --data_test $test_set --save_results --pretrain_cluster "${directory}/logs/dhp_restoration/edsr/edsr_scale4.pt" \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/edsr/edsr_cluster_scale4.pt"
#done


# 3. EDSR factor, sic = 3
MODEL=EDSR_FACTOR
N_BLOCK=8
N_FEATS=128
SIC=3
SCALE=4
export CHECKPOINT="Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}SIC${SIC}"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --sic_layer ${SIC} --res_scale 1 \
--input_dim 128 --data_test $test_set --save_results \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/edsr/edsr_factor_sic3_scale4.pt"
#done


# 4. EDSR factor, sic = 2
MODEL=EDSR_FACTOR
N_BLOCK=8
N_FEATS=128
SIC=2
SCALE=4
export CHECKPOINT="Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}SIC${SIC}"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --sic_layer ${SIC} --res_scale 1 \
--input_dim 128 --data_test $test_set --save_results \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/edsr/edsr_factor_sic2_scale4.pt"
#done


# 5. EDSR Basis, 128 + 40
MODEL=EDSR_Basis
N_BLOCK=8
N_FEATS=128
N_BASIS=40
S_BASIS=128
SCALE=4
export CHECKPOINT="Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_BaseS${S_BASIS}N${N_BASIS}"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --basis_size ${S_BASIS} --n_basis ${N_BASIS} --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --res_scale 1 \
--input_dim 128 --data_test $test_set --save_results \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/edsr/edsr_basis_128+40_scale4.pt"
#done


# 6. EDSR Basis, 128 + 27
MODEL=EDSR_Basis
N_BLOCK=8
N_FEATS=128
N_BASIS=27
S_BASIS=128
SCALE=4
export CHECKPOINT="Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_BaseS${S_BASIS}N${N_BASIS}"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --basis_size ${S_BASIS} --n_basis ${N_BASIS} --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --res_scale 1 \
--input_dim 128 --data_test $test_set --save_results \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/edsr/edsr_basis_128+27_scale4.pt"
#done


# 7. EDSR DHP, ratio = 0.2
MODEL=DHP_EDSR
N_BLOCK=8
N_FEATS=128
SCALE=4
RATIO=0.2
CHECKPOINT="Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_Ratio${RATIO}"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main_dhp.py --save $CHECKPOINT --model $MODEL --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --res_scale 1 \
--input_dim 128 --data_test $test_set --save_results --prune_upsampler \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/edsr/edsr_dhp_ratio2_scale4.pt"
#done


# 8. EDSR DHP, ratio = 0.4
MODEL=DHP_EDSR
N_BLOCK=8
N_FEATS=128
SCALE=4
RATIO=0.4
CHECKPOINT="Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_Ratio${RATIO}"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main_dhp.py --save $CHECKPOINT --model $MODEL --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --res_scale 1 \
--input_dim 128 --data_test $test_set --save_results --prune_upsampler \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/edsr/edsr_dhp_ratio4_scale4.pt"
#done


# 9. EDSR DHP, ratio = 0.6
MODEL=DHP_EDSR
N_BLOCK=8
N_FEATS=128
SCALE=4
RATIO=0.6
CHECKPOINT="Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_Ratio${RATIO}"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main_dhp.py --save $CHECKPOINT --model $MODEL --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --res_scale 1 \
--input_dim 128 --data_test $test_set --save_results --prune_upsampler \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/edsr/edsr_dhp_ratio6_scale4.pt"
#done
done

