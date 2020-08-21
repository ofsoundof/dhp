#!/bin/bash
#Submit to GPU


#for T in 2e-3 5e-3; do
for RATIO in 0.2 0.4 0.6; do
MODEL=UnetDN5_DHP
SIGMA=70
N_PATCH=128
N_BATCH=16
REG=1e-1
T=5e-3
LIMIT=0.02
#RATIO=0.8
CHECKPOINT="${MODEL}_Sigma${SIGMA}_P${N_PATCH}B${N_BATCH}_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}"
echo $CHECKPOINT

qsub -N $CHECKPOINT ./vpython_kaizhang.sh ../main_dhp.py --save $CHECKPOINT --model $MODEL --n_feats 32 --n_colors 1 \
--patch_size $N_PATCH --batch_size $N_BATCH --noise_sigma ${SIGMA} --scale 1 \
--epochs 60 --lr_decay_step 10+40 --test_every 10000 --n_threads 6 --data_train DIV2KDENOISE --data_test DenoiseSet68 --ext bin \
--prune_threshold ${T} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} \
--save_results --print_model --reset --input_dim 128 



MODEL=DNCNN_DHP
SIGMA=70
N_PATCH=64
N_BATCH=64
REG=1e-1
T=5e-3
LIMIT=0.02
#RATIO=0.8
CHECKPOINT="${MODEL}_Sigma${SIGMA}_P${N_PATCH}B${N_BATCH}_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}"
echo $CHECKPOINT

qsub -N $CHECKPOINT ./vpython_kaizhang.sh ../main_dhp.py --save $CHECKPOINT --model $MODEL --m_blocks 15 --n_feats 64 --n_colors 1 --bn True \
--patch_size $N_PATCH --batch_size $N_BATCH --noise_sigma $SIGMA --scale 1 \
--epochs 60 --lr_decay_step 10+40 --test_every 10000 --n_threads 6 --data_train DIV2KDENOISE --data_test DenoiseSet68 --ext bin \
--prune_threshold ${T} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} \
--save_results --print_model --reset --input_dim 128

done
#done



MODEL=UnetDN5
SIGMA=70
N_PATCH=128
N_BATCH=16
CHECKPOINT="${MODEL}_Sigma${SIGMA}_P${N_PATCH}_B${N_BATCH}"
echo $CHECKPOINT

qsub -N $CHECKPOINT ./vpython_kaizhang.sh ../main.py --save ${CHECKPOINT} --model $MODEL --n_feats 32 --n_colors 1 \
--patch_size ${N_PATCH} --batch_size ${N_BATCH} --noise_sigma $SIGMA  --scale 1 \
--epochs 60 --lr_decay 40 --test_every 10000 --n_threads 6 --data_train DIV2KDENOISE --data_test DenoiseSet68 --ext bin \
--save_results --print_model --reset --input_dim 128 



MODEL=DNCNN
SIGMA=70
N_PATCH=64
N_BATCH=64
CHECKPOINT="${MODEL}_Sigma${SIGMA}_P${N_PATCH}_B${N_BATCH}"
echo $CHECKPOINT

qsub -N $CHECKPOINT ./vpython_kaizhang.sh ../main.py --save $CHECKPOINT --model $MODEL --m_blocks 15 --n_feats 64 --n_colors 1 --bn True \
--patch_size $N_PATCH --batch_size $N_BATCH --noise_sigma $SIGMA --scale 1 \
--epochs 60 --lr_decay 40 --test_every 10000 --n_threads 6 --data_train DIV2KDENOISE --data_test DenoiseSet68 --ext bin \
--save_results --print_model --reset --input_dim 128











