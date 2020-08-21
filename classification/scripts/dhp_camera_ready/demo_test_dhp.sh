#!/bin/bash
#Submit to GPU

directory=~/projects/
dir_pretrain="${directory}/image_classification/dhp/classification/model_zoo"
dir_data="${directory}/data/"
dir_save="${directory}/logs/dhp_results/"

#################################
# MobileNet
#################################

# MobileNet, Ratio=0.10
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save MobileNet_DHP_Ratio10 --template Tiny_ImageNet --model MobileNet_DHP --test_only \
--pretrain "${dir_pretrain}/mobilenetv1_ratio10.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# MobileNet, Ratio=0.30
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save MobileNet_DHP_Ratio30 --template Tiny_ImageNet --model MobileNet_DHP --test_only \
--pretrain "${dir_pretrain}/mobilenetv1_ratio30.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# MobileNet, Ratio=0.50
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save MobileNet_DHP_Ratio50 --template Tiny_ImageNet --model MobileNet_DHP --test_only \
--pretrain "${dir_pretrain}/mobilenetv1_ratio50.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# MobileNet, Ratio=0.90
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save MobileNet_DHP_Ratio90 --template Tiny_ImageNet --model MobileNet_DHP --test_only \
--pretrain "${dir_pretrain}/mobilenetv1_ratio90.pt" --dir_data  ${dir_data} --dir_save ${dir_save}


#################################
# MobileNetV2
#################################

# MobileNetV2, Ratio=0.10
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save MobileNetV2_DHP_Ratio10 --template Tiny_ImageNet --model MobileNetV2_DHP --test_only \
--pretrain "${dir_pretrain}/mobilenetv2_ratio10.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# MobileNetV2, Ratio=0.30
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save MobileNetV2_DHP_Ratio30 --template Tiny_ImageNet --model MobileNetV2_DHP --test_only \
--pretrain "${dir_pretrain}/mobilenetv2_ratio30.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# MobileNetV2, Ratio=0.50
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save MobileNetV2_DHP_Ratio50 --template Tiny_ImageNet --model MobileNetV2_DHP --test_only \
--pretrain "${dir_pretrain}/mobilenetv2_ratio50.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# MobileNetV2, Ratio=0.90
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save MobileNetV2_DHP_Ratio90 --template Tiny_ImageNet --model MobileNetV2_DHP --test_only \
--pretrain "${dir_pretrain}/mobilenetv2_ratio90.pt" --dir_data  ${dir_data} --dir_save ${dir_save}


#################################
# ResNet20
#################################

# ResNet20, Ratio=0.5
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L20_Ratio50 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 20 --test_only \
--pretrain "${dir_pretrain}/resnet20_ratio50.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

#################################
# ResNet50
#################################

# ResNet56, Ratio=0.38
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L56_Ratio38 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 56 --test_only \
--pretrain "${dir_pretrain}/resnet56_ratio38.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# ResNet56, Ratio=0.5
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L56_Ratio50 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 56 --test_only \
--pretrain "${dir_pretrain}/resnet56_ratio50.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

#################################
# ResNet110
#################################

# ResNet110, Ratio=0.1
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L110_Ratio10 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 110 --test_only \
--pretrain "${dir_pretrain}/resnet110_ratio10.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# ResNet110, Ratio=0.2
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L110_Ratio20 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 110 --test_only \
--pretrain "${dir_pretrain}/resnet110_ratio20.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# ResNet110, Ratio=0.38
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L110_Ratio38 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 110 --test_only \
--pretrain "${dir_pretrain}/resnet110_ratio38.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# ResNet110, Ratio=0.5
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L110_Ratio50 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 110 --test_only \
--pretrain "${dir_pretrain}/resnet110_ratio50.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# ResNet110, Ratio=0.62
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L110_Ratio62 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 110 --test_only \
--pretrain "${dir_pretrain}/resnet110_ratio62.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# ResNet110, Ratio=0.7
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L110_Ratio70 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 110 --test_only \
--pretrain "${dir_pretrain}/resnet110_ratio70.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

#################################
# ResNet164
#################################

# ResNet164, Ratio=0.1
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L164_Ratio10 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 164 --test_only \
--pretrain "${dir_pretrain}/resnet164_ratio10.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# ResNet164, Ratio=0.2
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L164_Ratio20 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 164 --test_only \
--pretrain "${dir_pretrain}/resnet164_ratio20.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# ResNet164, Ratio=0.38
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L164_Ratio38 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 164 --test_only \
--pretrain "${dir_pretrain}/resnet164_ratio38.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# ResNet164, Ratio=0.5
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L164_Ratio50 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 164 --test_only \
--pretrain "${dir_pretrain}/resnet164_ratio50.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# ResNet164, Ratio=0.62
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L164_Ratio62 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 164 --test_only \
--pretrain "${dir_pretrain}/resnet164_ratio62.pt" --dir_data  ${dir_data} --dir_save ${dir_save}

# ResNet164, Ratio=0.7
CUDA_VISIBLE_DEVICES=1 python ../../main_dhp.py --save ResNet_DHP_SHARE_L164_Ratio70 --template CIFAR10_ResNet --model ResNet_DHP_SHARE --depth 164 --test_only \
--pretrain "${dir_pretrain}/resnet164_ratio70.pt" --dir_data  ${dir_data} --dir_save ${dir_save}



