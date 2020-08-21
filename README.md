# [DHP: Differentiable Meta Pruning via HyperNetworks](https://arxiv.org/abs/2003.13683)
This is the official implementation of "DHP: Differentiable Meta Pruning via HyperNetworks".

## Contents
1. [Introduction](#introduction)
2. [Contribution](#contribution)
3. [Methodology](#methodology)
4. [Dependencies](#dependencies)
5. [Image Classification](#image-classification)
6. [Image Restoration](#image-restoration)
7. [Results](#results)
8. [Reference](#reference)
9. [Acknowledgements](#acknowledgements)

## Introduction
Network pruning has been the driving force for the acceleration of neural networks and the alleviation of model storage/transmission burden. With the advent of AutoML and neural architecture search (NAS), pruning has become topical with automatic mechanism and searching based architecture optimization. Yet, current automatic designs rely on either reinforcement learning or evolutionary algorithm. Due to the non-differentiability of those algorithms, the pruning algorithm needs a long searching stage before reaching the convergence.

To circumvent this problem, this paper introduces a differentiable pruning method via hypernetworks for automatic network pruning. The specifically designed hypernetworks take latent vectors as input and generate the weight parameters of the backbone network. The latent vectors control the output channels of the convolutional layers in the backbone network and act as a handle for the pruning of the layers. By enforcing ℓ1 sparsity regularization to the latent vectors and utilizing proximal gradient solver, sparse latent vectors can be obtained. Passing the sparsified latent vectors through the hypernetworks, the corresponding slices of the generated weight parameters can be removed, achieving the effect of network pruning. The latent vectors of all the layers are pruned together, resulting in an automatic layer configuration. Extensive experiments are conducted on various networks for image classification, single image super-resolution, and denoising. And the experimental results validate the proposed method. 

## Contribution
**1. A new architecture of hypernetwork is designed. Different from the classicalhypernetwork composed of linear layers, the new design is tailored to automatic network pruning. By only operating on the input of the hypernetwork,the backbone network can be pruned.**
**2. A differentiable automatic networking pruning method is proposed. The differentiability comes with the designed hypernetwork and the utilized proximal gradient. It accelerates the convergence of the pruning algorithm.**
**3. By the experiments on various vision tasks and modern convolutional neural  networks  (CNNs),  the  potential  of  automaticnetwork pruning as fine-grained architecture search is revealed.**

## Methodology
<img src="/figs/dhp_pipeline.png" width="900">

The workflow of the proposed differentiable pruning method. The latent vectors **z** attached  to  the  convolutional  layers  act  as  the  handle  for  network  pruning.  The hypernetwork  takes  two  latent  vectors  as  input  and  emits  output  as  the  weight  ofthe  backbone  layer. l1 sparsity  regularization  is  enforced  on  the  latent  vectors.  The differentiability  comes  with  the  hypernetwork  tailored  to  pruning  and  the  proximal gradient exploited to solve problem. After the pruning stage, sparse latent vectors are obtained which result in pruned weights after being passed through the hypernetwork

<img src="/figs/hypernetwork.png" width="500">

Illustration  of  the  hypernetwork  designed  for  network  pruning.  It  generates a  weight  tensor  after  passing  the  input  latent  vector  through  the  latent  layer,  theembedding layer, and the explicit layer. If one element in the latent vector of the current layer is pruned, the corresponding slice of the output tensor is also pruned.

## Dependencies
* Python 3.7.4
* PyTorch >= 1.2.0
* numpy
* matplotlib
* tqdm
* scikit-image
* easydict
* IPython

## Image Classification


## Image Restoration

 
## Results

<img src="/figs/hinge_kse_flops.png" width="400"> <img src="/figs/hinge_kse_params.png" width="400">

FLOP and parameter comparison between KSE and Hinge under different compression ratio. ResNet56 is compressed. Top-1 error rate is reported.

<img src="/figs/resnet164_cifar100.png" width="400"> <img src="/figs/resnext164_cifar100.png" width="400">

Comparison between SSS and the proposed Hinge method on ResNet and ResNeXt. Top-1 error rate is reported for CIFAR100.

<img src="/figs/table1.png" width="450">
<img src="/figs/table2.png" width="450">
<img src="/figs/table4.png" width="450">


## Reference
If you find our work useful in your research of publication, please cite our work:

```
@inproceedings{li2020group,
  title={Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression},
  author={Li, Yawei and Gu, Shuhang and Mayer, Christoph and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2020}
}
```

## Acknowledgements
This work was partly supported by the ETH Zurich Fund (OK), by VSS ASTRA, SBB and Huawei projects, and by Amazon AWS and Nvidia GPU grants.

This repository is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for making their EDSR codes public.

This repository is also based on the [implementation](https://github.com/ofsoundof/learning_filter_basis) of our former paper [Learning Filter Basis for Convolutional Neural Network Compression](https://arxiv.org/abs/1908.08932). If you are interested, please refer to:

```
@inproceedings{li2019learning,
  title = {Learning Filter Basis for Convolutional Neural Network Compression},
  author = {Li, Yawei and Gu, Shuhang and Van Gool, Luc and Timofte, Radu},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
  year = {2019}
}
```

