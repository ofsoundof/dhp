#!/bin/bash
#Submit to GPU


directory=~/projects

for test_set in Set5 Set14 B100 Urban100 DIV2K; do
# 1. SRResNet Original
MODEL=SRRESNET
N_BLOCK=16
N_FEATS=64
SCALE=4
export CHECKPOINT=" Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS}  \
--input_dim 128 --data_test $test_set --save_results \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/srresnet/srresnet_scale4.pt"
#done


# 2. SRResNet cluster
#for test_set in Set5; do
TEMPLATE=SRRESNET_CLUSTER
N_BLOCK=16
N_FEATS=64
SCALE=4
CHECKPOINT=" Test/${TEMPLATE}_X${SCALE}_L${N_BLOCK}F${N_FEATS}"
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --template $TEMPLATE --model ClusterNet --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} \
--input_dim 128 --data_test $test_set --save_results --pretrain_cluster "${directory}/logs/dhp_restoration/srresnet/srresnet_scale4.pt" \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/srresnet/srresnet_cluster_scale4.pt"
#done


# 3. SRResNet factor, sic = 3
MODEL=SRRESNET_FACTOR
N_BLOCK=16
N_FEATS=64
SIC=3
SCALE=4
export CHECKPOINT=" Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}SIC${SIC}"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --sic_layer ${SIC} \
--input_dim 128 --data_test $test_set --save_results \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/srresnet/srresnet_factor_sic3_scale4.pt"
#done


# 4. SRResNet factor, sic = 2
MODEL=SRRESNET_FACTOR
N_BLOCK=16
N_FEATS=64
SIC=2
SCALE=4
export CHECKPOINT=" Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}SIC${SIC}"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --sic_layer ${SIC} \
--input_dim 128 --data_test $test_set --save_results \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/srresnet/srresnet_factor_sic2_scale4.pt"
#done


# 5. SRResNet basis, 32+32
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --model SRResNet_Basis --save  Test/SRResNet_Basis_X4_L16_B32+32 --scale 4 --basis_size 32 --n_basis 32 --bn_every --n_resblocks 16 --n_feats 64 \
--input_dim 128 --data_test $test_set --save_results \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/srresnet/srresnet_basis_32+32_scale4.pt"
#done


# 6. SRResNet basis, 64+14
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main.py --model SRResNet_Basis --save  Test/SRResNet_Basis_X4_L16_B64+14 --scale 4 --basis_size 64 --n_basis 14 --bn_every --n_resblocks 16 --n_feats 64 \
--input_dim 128 --data_test $test_set --save_results \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/srresnet/srresnet_basis_64+14_scale4.pt"
#done


#<<COMMENT
# 7. SRResNet DHP, ratio = 0.2
MODEL=DHP_SRResNet
N_BLOCK=16
N_FEATS=64
SCALE=4
RATIO=0.2
CHECKPOINT=" Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_Ratio${RATIO}_Upsampler"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main_dhp.py --save $CHECKPOINT --model $MODEL --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} \
--input_dim 128 --data_test $test_set --save_results --prune_upsampler \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/srresnet/srresnet_dhp_ratio2_scale4.pt"
#done


# 8. SRResNet DHP, ratio = 0.4
MODEL=DHP_SRResNet
N_BLOCK=16
N_FEATS=64
SCALE=4
RATIO=0.4
CHECKPOINT=" Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_Ratio${RATIO}_Upsampler"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main_dhp.py --save $CHECKPOINT --model $MODEL --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} \
--input_dim 128 --data_test $test_set --save_results --prune_upsampler \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/srresnet/srresnet_dhp_ratio4_scale4.pt"
#done


# 9. SRResNet DHP, ratio = 0.6
MODEL=DHP_SRResNet
N_BLOCK=16
N_FEATS=64
SCALE=4
RATIO=0.6
CHECKPOINT=" Test/${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_Ratio${RATIO}_Upsampler"
echo $CHECKPOINT
#for test_set in Set5; do
CUDA_VISIBLE_DEVICES=1 python ../main_dhp.py --save $CHECKPOINT --model $MODEL --scale $SCALE --n_resblocks $N_BLOCK --n_feats ${N_FEATS} \
--input_dim 128 --data_test $test_set --save_results --prune_upsampler \
--dir_save "${directory}/logs" \
--dir_data "${directory}/data" \
--test_only --pretrain "${directory}/logs/dhp_restoration/srresnet/srresnet_dhp_ratio6_scale4.pt"
#done
done












