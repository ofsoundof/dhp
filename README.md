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
<img src="/figs/dhp_pipeline.png" width="700">

The workflow of the proposed differentiable pruning method. The latent vectors **z** attached  to  the  convolutional  layers  act  as  the  handle  for  network  pruning.  The hypernetwork  takes  two  latent  vectors  as  input  and  emits  output  as  the  weight  ofthe  backbone  layer. l1 sparsity  regularization  is  enforced  on  the  latent  vectors.  The differentiability  comes  with  the  hypernetwork  tailored  to  pruning  and  the  proximal gradient exploited to solve problem. After the pruning stage, sparse latent vectors are obtained which result in pruned weights after being passed through the hypernetwork

<img src="/figs/hypernetwork.png" width="700">

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
For the details on image classification, please refer to [classification](/classification).

## Image Restoration
For the details on image restoration, please refer to [restoration](/restoration).
 
## Results

<img src="/figs/table1.png" width="600"> 

<img src="/figs/table2.png" width="600">

<img src="/figs/table3.png" width="600">

<img src="/figs/figure.png" width="600">


## Reference
If you find our work useful in your research of publication, please cite our work:

```
@inproceedings{li2020dhp,
  title={DHP: Differentiable Meta Pruning via HyperNetworks},
  author={Li, Yawei and Gu, Shuhang and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}
```

## Acknowledgements
This work was partly supported by the ETH Zurich Fund (OK), a Huawei Tech-nologies Oy (Finland) project, an Amazon AWS grant, and an Nvidia grant.

This repository is based on the our former paper [Filter Basis](https://github.com/ofsoundof/learning_filter_basis) and [Group Sparsity](https://github.com/ofsoundof/group_sparsity). If you are interested, please refer to:

```
@inproceedings{li2020group,
  title={Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression},
  author={Li, Yawei and Gu, Shuhang and Mayer, Christoph and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2020}
}

@inproceedings{li2019learning,
  title = {Learning Filter Basis for Convolutional Neural Network Compression},
  author = {Li, Yawei and Gu, Shuhang and Van Gool, Luc and Timofte, Radu},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
  year = {2019}
}
```

