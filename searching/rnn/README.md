## AutoShrink: RNN Architecture Search Instructions

This README contains the instructions for RNN architecture search using the proposed AutoShrink method. Our reported search cost is measured on NVIDIA TITAN Xp, and the search cost will be lower using modern GPUs like NVIDIA TITAN RTX.

#### Dependencies

```pip
Anaconda Essentials (e.g. NumPy)
matplotlib >= 3.0.0
pytorch >= 1.2.0
```

#### Understand the code

`autoshrink_rnn.py`: The framework of exploring and building RNN networks with AutoShrink strategy.

`data.py`: Data utilities for Penn Treebank dataset.

`model.py`: RNN model builder.

`preprocess.py`: Preprocessing data/embedding.

`train.py`: Main application for AutoShrink method to search for efficient RNN architectures.

`utils.py`: Data processing utilities.

#### Searching for RNN architectures

Efficient but powerful RNN architectures can be found within 1.5 GPU hours by running the script below:

```sh
python searching/rnn/train.py \
--nodes 6 \														# Number of nodes in the search space
--data ./data/penn_proxy/ \									# Proxy Dataset
--emb_path ./searching/rnn/w2v.pkl \  # Word to vector word embedding
--emsize 100 \												# Embedding size
--nhidlast 100 \											# Number of hidden units for last layer
--lr 20 \															# Learning rate
--clip 0.25 \													# Gradient Clipping
--epochs 10 \													# Proxy evaluation epochs
--batch_size 128											# Batch size
```

Similar to CNN architecture search, the best RNN architecture can be explored by checking the `best_perf_log.png`. Architecture details can be checked by unpickling the corresponding graph stored using the following scripts:

```python
import pickle
with open("graph-best", "rb") as fp:
  graph = pickle.load(fp)
# Show the mask after the autoshrink process.
print(graph.mask)
# Show the search space
print(graph.ops_def)
```

And check the properties of the object.

**WARNING**: Due to the random characteristics of NAS algorithms, this script does not gaurantee to find exactly the same architecture as is shown in the paper. However, you can expect to find models with performance on the same competitive level.