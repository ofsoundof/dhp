
## Test
1. `pip install -r ../requirements.txt`

2. Download [Tiny-ImageNet]{} from [Google Drive] or [Dropbox]. CIFAR dataset is automatically downloaded the first time the code is run. Place the dataset at your `--dir_data` directory.

2. Download the model zoo from [Google Drive](https://drive.google.com/file/d/1ojU4jkgwJ6-qHlbD1e_nBWlUAt6ouNmz/view?usp=sharing) or [Dropbox](https://www.dropbox.com/s/vrc3zacctwm3z11/model_zoo_classification.zip?dl=0). This contains the compressed models. Place the models in `./model_zoo`.

3. [`cd ./scripts/dhp_camera_ready`](./scripts/dhp_camera_ready). 

4. Use the following scripts in [`./scripts/dhp_camera_ready/demo_test_dhp.sh`](./scripts/dhp_camera_ready/demo_test_dhp.sh) to test the compressed models. 

    Be sure the change the directories `--pretrain`, `--dir_data`, and `--dir_save`.

    `--pretrain`: the pretrained model.

	`--dir_data`: where the dataset is stored.

	`--dir_save`: where you want to save the results.
5. Demo
```bash
	directory=~/projects/
	dir_pretrain="${directory}/***/model_zoo"
	dir_data=***
	dir_save="***
	# ResNet56, Ratio=0.5
	CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L56_Ratio50 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 56 --test_only \
	--pretrain "${dir_pretrain}/resnet56_ratio50.pt" --dir_data  ${dir_data} --dir_save ${dir_save}
```
## Train

1. [`cd ./scripts/dhp_camera_ready`](./scripts/dhp_camera_ready). 

2. Run the scripts `dhp_XXX.sh` to reproduce the results in our paper, where `XXX` may be replace by `mobilenet`, `mobilenetv2`, `resnet20`, `resnet56`, `resnet110` and `resnet164` depending on which network you want to compress. 

3. Be sure the change the directories `--pretrain`, `--dir_data`, and `--dir_save`.
