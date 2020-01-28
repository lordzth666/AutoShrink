## AutoShrink: CNN Architecture Search Instructions

This README contains the instructions for CNN architecture search using the proposed AutoShrink method. Our reported search cost is measured on NVIDIA TITAN Xp, and the search cost will be lower using modern GPUs like NVIDIA TITAN RTX.

#### Dependencies
```pip
Anaconda Essentials (e.g. NumPy)
tensorflow-gpu >= 1.13.0
matplotlib >= 3.0.0
networkx >= 2.4
opencv >= 3.4
tqdm >= 4.38
```
Should there be any missing package, please use `pip` or `conda` to install.

#### Data preparation
To make the CIFAR-10 proxy dataset, one can run the following script:
```sh
python searching/cnn/make_cifar10_proxy.py \
--record_dir ./data
```

#### Network Specification
Network specifications are stored in `.prototxt` files, similar to the old-fashioned `caffe` style. `.prototxt` files, as well as `.solver` files, are parsed and compiled using `api/network/model.py`. Please check the implementation for details.

#### Search Space

The search space for CNN architectures is composed of Directed Acyclic Graphs (DAGs). DNN operation are represented as nodes, and the distributions of tensors are represented as edge.

#### Understand the code

`autoshrink_cnn.py`: The main application to start AutoShrink searching.

`metagraph.py`: Search space construction. AutoShrink is implemented here.

`ops_gen.py`: Search space specification.

`proxy_estimator`: Estimator of performance on the proxy dataset.

`util_concat`: Functional tools (utilities) for `metagraph.py`.

`writer.py`: Writer to construct the prototxt.

#### Kick off AutoShrink CNN searching

A common configuration as is proposed in the paper is to use `N=8` nodes in the initial DAG. `k=10` is used in the **k-candidate-selection** strategy to have the best trade-off between search cost and the quality of explored cells.

```sh
CUDA_VISIBLE_DEVICES=0 python searching/cnn/autoshrink_cnn.py \
--model_root ./autoshrink \
--nodes 8 \												# 8 nodes in a DAG
--depth 3 \												# 3 stages in total
--width 1 \												# 1 DAG in parallel per stage.
--scale 1 2 4 \										# Scale factor (channel width multiplier) for each stage
--pool 1 2 2 \										# Pooling size specification for each stage
--replicate 1 1 1 \								# Number of cells stacked in each stage
--protocol v3 \										# Search space to use.
--max_steps 10										# k in 'k-candidate selection strategy'
```

Upon success completion of the training process, a list of 'graphs' can be retrieved in the `model_root` folder . Please refer to `best_perf.log` to select the graph with the best metric score (search objective) as the best cell topology in the AutoShrink algorithm.

**WARNING**

* Due to the random characteristics of NAS, this code **DO NOT** gaurantee the exploration of exactly the same architectures as is proposed in the paper. However, architectures with equally strong competence can be found using this script.
* As a tie-breaker, architectures with smaller number of branches may be our preference, as training these architectures will require significant smaller amount of time.

#### Building a ShrinkCNN according to best cell topology

Upon choosing the best cell topology, ShrinkCNNs can be built upon the following scripts:

```sh
# Please change this before proceeding.
BEST_GRAPH_NAME=./autoshrink_cnn_demo/graph-0.29

CUDA_VISIBLE_DEVICES=0,1 python evaluation/cnn/build_imagenet.py \
--graph_path $BEST_GRAPH_NAME \			# best graph
--scale 0.5 1 2 4 8 \								# Scale multiplier for each stage
--channel_scale 1.4 \								# Width multiplier
--replicate 1 3 4 6 4 \							# number of cells to stack in each stage
--regularizer 5e-5 \								# regularizer
--pool_size 1 2 2 2 2 \							# Pool size for each stage
--use_residual \										# Use residual connection to connect each cell
--image_size 224										# Train with 224x224 images
```

Usually, the built architecture have a Multiply-Accumulates Cost of 300M~500M. If you find that your architecture exceeds this computation budget, please consider decreasing the `scale` and/or `channel_scale` in the script above.