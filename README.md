# AutoShrink: A Topology-aware NAS for Discovering Efficient Neural Architecture

### Paper

Tunhou Zhang, Hsin-Pai Cheng, Zhenwen Li, Feng Yan, Chengyu Huang, Hai Li, and Yiran Chen. "AutoShrink: A Topology-aware NAS for Discovering Efficient Neural Architecture."  arXiv Link: https://arxiv.org/abs/1911.09251

### Description

This is the official implementation of our AAAI'20 paper: **AutoShrink: A Topology-aware NAS for Discovering Efficient Neural Architecture**. AutoShrink targets on efficient neural architecture search within only **1.5 GPU hours**, which is at least 6.7x faster than the fastest existing NAS methods (ENAS). Moreover, AutoShrink faciliates architecture search for both CNN and RNN architectures, leading to efficient architectures of comparable accuracy of the SOTA models.

![Workflow](https://github.com/lordzth666/AutoShrink/raw/master/g3doc/workflow3.png)

### Understand the code

The structure of this project is shown as follows:

```sh
├── api
|   ├── dataset
|   └── network
|   └── ...
├── data
|   ├── ...
├── evaluation
|   ├── cnn
|		   ├── base_estimator.py
|		   ├── ...
|   └── rnn
|		   ├── model.py
|		   ├── ...
├── searching
|   ├── cnn
|		   ├── solver
|		   ├── autoshrink_cnn.py
|		   ├── ...
|   └── rnn
|		   ├── autoshrink_rnn.py
|		   ├── ...
├── README.md
```

`api`: This is the low-level implementation of layers, training schemes etc. for the ShrinkCNN-series networks and CIFAR/ImageNet dataset.

`data`: This folder is used to hold the training dataset (i.e. CIFAR, ImageNet, Penn-Treebank etc.).

`evaluation`: This folder contains the evaluation code of the derived architectures from AutoShrink. Please refer to `README.md` inside for more detailed instructions.

`searching`: This folder contains the searching code of the AutoShrink. The subfolder `cnn` and `rnn` corresponds to CNN and RNN architecture search, respectively. Please refer to `README.md` inside for more detailed instructions.

### Experimental Results

**ShrinkCNNs**

ShrinkCNNs are crafted within only 1.5 GPU hours. ~75.1%(73.9%) top-1 accuracy can be achieved using Swish/ReLU activations.

| Architecture        | Top-1 Accuracy | \#Parameters | \#Multiply-Accumulates |
| ------------------- | -------------- | ------------ | ---------------------- |
| ShrinkCNN-A (ReLU)  | 73.9%          | 3.6M         | 384M                   |
| ShrinkCNN-B (Swish) | 75.1%          | 3.6M         | 384M                   |

**ShrinkRNNs**

ShrinkRNNs are crafted within only 1.5 GPU hours.

| Architecture | Perplexity(valid) | Perplexity(test) | \#Parameters |
| ------------ | ----------------- | ---------------- | ------------ |
| ShrinkRNN    | 58.5              | 56.5             | 24M          |

The implementation of ShrinkRNN is in part intuited by the [DARTS project](https://github.com/quark0/darts) (Liu et al.).

### Citation

If you would like to use any part of this code for research, please kindly cite our paper:

```citation
@inproceedings{zhang2019autoshrink,
title={AutoShrink: A Topology-aware NAS for Discovering Efficient Neural Architecture},
author={Zhang, Tunhou and Cheng, Hsin-Pai and Li, Zhenwen and Yan, Feng and Huang, Chengyu and Li, Hai and Chen, Yiran},
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2020)},
year={2019}
}
```
