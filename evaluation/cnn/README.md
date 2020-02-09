## AutoShrink: CNN Architecture Evaluation Instructions

This README contains the instructions for serving the ShrinkCNN models in the ImageNet-1k classification tasks. The training process will take 300 epochs to reach convergence (~600 GPU hours with NVIDIA TITAN RTX). A cosine learning rate decay scheduler will be applied to best optimize the cell structure.

#### Dependencies
```pip
Anaconda Essentials (e.g. NumPy)
tensorflow-gpu >= 1.13.0
matplotlib >= 3.0.0
networkx >= 2.4
opencv >= 3.4
tqdm >= 4.38
```

#### Data preparation
Please refer to [TensorFlow Models repository] (https://github.com/tensorflow/models/tree/master/research/inception/inception/data)
for detailed instructions for building the ImageNet TFRecords.
The directory layout for the training images should be as follows:
```sh
├── ImagenetTF
|   ├── train_tfrecord
|		   ├── train-00000-of-01024
|		   ├── train-00001-of-01024
|		   ├── ...
|		   ├── train-01023-of-01024
|   └── val_tfrecord
|		   ├── validation-00000-of-00128
|		   ├── validation-00001-of-00128
|		   ├── ...
|		   ├── validation-00127-of-00128
```
#### Train the ShrinkCNNs

ShrinkCNN-A(B) can achieve ~73.9(75.1) top-1 accuracy using the scripts below:

```sh
CUDA_VISIBLE_DEVICES=0,1 python evaluation/cnn/imagenet-eval.py \
--proto ./evaluation/cnn/shrinkCNN-A-release-repro.prototxt \
--batch_size 128 \
--ngpus 2 \
--epochs 300 \
--solver ./evaluation/cnn/solver/imagenet.solver
```

As exponential moving average is applied, please run the following scripts for evaluating the results:
```sh
CUDA_VISIBLE_DEVICES=0 python evaluation/cnn/imagenet-valid.py \
--proto ./evaluation/cnn/shrinkCNN-A-release-repro.prototxt \
--ngpus 1 \
--solver ./evaluation/cnn/solver/imagenet.solver
```


#### Hyperparameter Settings

The experiments use the following hyperparameter settings:

```c
mini_batch_size: 128
total_batch_size: 256
reference_lr: 0.1
moving_average_decay: 0.999
BN_momentum: 0.999
BN_epsilon: 1e-3
scheduler: 'cosine_decay'
L2-regularization strength: 4e-5
Dropout(Before FC): 0.0
Epochs: 300
```

All of the above settings are served in `evaluation/solver/imagenet.solver` and the implenetations in the `api` folder. You may change them as you wish.

**Note**

* The multi-gpu strategy will automatically scale up the learning rate linearly. Therefore, learning rate in the `imagenet.solver` should be **0.1/2** rather than **0.1**.

* Note that to achieve the best result, additional cosine annealing strategy may be applied to boost accuracy.

* Results may vary from time to time, due to the nature of NAS networks.
