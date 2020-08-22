
## Test
1. `pip install -r ../requirements.txt`

2. Download [Tiny-ImageNet](http://www.image-net.org/) from [Google Drive](https://drive.google.com/file/d/1Aajaob10vzDqPbWZVNMYxJRowlNE0yd5/view?usp=sharing) or [Dropbox](https://www.dropbox.com/s/2kbqse543y2ule0/tiny-imagenet-200.tar.xz?dl=0). CIFAR dataset is automatically downloaded the first time the code is run. Place the dataset at your `--dir_data` directory.

3. Download the model zoo from [Google Drive](https://drive.google.com/file/d/1ojU4jkgwJ6-qHlbD1e_nBWlUAt6ouNmz/view?usp=sharing) or [Dropbox](https://www.dropbox.com/s/vrc3zacctwm3z11/model_zoo_classification.zip?dl=0). This contains the compressed models. Place the models in `./model_zoo`.

4. [`cd ./scripts/dhp_camera_ready`](./scripts/dhp_camera_ready). 

5. Use the following scripts in [`./scripts/dhp_camera_ready/demo_test_dhp.sh`](./scripts/dhp_camera_ready/demo_test_dhp.sh) to test the compressed models. 

    Be sure the change the directories `--pretrain`, `--dir_data`, and `--dir_save`.

    `--pretrain`: where the pretrained models are placed.

    `--dir_data`: where the dataset is stored.

    `--dir_save`: where you want to save the results.
6. Demo: test ResNet56 with target compression ratio at about 50%.
```bash
	# ResNet56, Ratio=0.5
	python ../../main_dhp.py --save ResNet_DHP_SHARE_L56_Ratio50 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 56 --test_only \
	--pretrain XXX --dir_data  XXX --dir_save XXX
```
## Train

1. [`cd ./scripts/dhp_camera_ready`](./scripts/dhp_camera_ready). 

2. Run the scripts `dhp_XXX.sh` to reproduce the results in our paper, where `XXX` may be replaced by `mobilenet`, `mobilenetv2`, `resnet20`, `resnet56`, `resnet110` and `resnet164` depending on which network you want to compress. 

3. Be sure the change the directories `--dir_data` and `--dir_save`.

4. Demo: compress ResNet56 with target compression ratio 50%.
```bash
	# ResNet56, Ratio=0.50
	MODEL=ResNet_DHP_SHARE
	LAYER=56
	BATCH=64
	TEMPLATE=CIFAR10
	REG=3e-4
	T=5e-3
	LIMIT=0.01
	RATIO=0.5
	CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}_B${BATCH}_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}
	python ../../main_dhp.py --save $CHECKPOINT --template "${TEMPLATE}_ResNet" --model ${MODEL} --batch_size ${BATCH} --epochs 300 --decay step-20-50+step-150-225 \
	--depth ${LAYER} --prune_threshold ${T} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} --print_model \
	--dir_save XXX --dir_data XXX
```
