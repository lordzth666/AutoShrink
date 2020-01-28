## AutoShrink: RNN Architecture Evaluation Instructions

This README contains the instructions for serving the ShrinkRNN models in the Penn-Treebank task.

#### Dependencies
```pip
Anaconda Essentials (e.g. NumPy)
matplotlib >= 3.0.0
pytorch >= 1.2.0
```

#### Train the ShrinkRNN

ShrinkRNN can achieve ~58.5/56.5 ppl on the validation/test dataset of the Penn-Treebank dataset,
using the script below:

```sh
CUDA_VISIBLE_DEVICES=0 python evaluation/rnn/test_final.py
```

#### Hyperparameter Settings

The experiments use the following hyperparameter settings:

```c
lr: 20
batch_size: 64
Epochs: 3000
Dropout_layer: 0.75
Dropout_hidden: 0.25
Dropout_input: 0.75
Dropout_embedding: 0.2
Dropout_words_embedding: 0.1 (Dropout to remove words from embedding layer)
L2-weight decay: 8e-7
```

We also use ASGD optimizer to improve the performance.

**Note**

* Results may vary from time to time while running this script, due to the nature of NAS networks.
