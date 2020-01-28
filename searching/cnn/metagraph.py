"""
Copyright (c) <2019> <CEI Lab, Duke University>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf
import numpy as np
from searching.cnn import util_concat as util
from searching.cnn import ops_gen
from searching.cnn.writer import ProtoWriter

from api.proto.layer import Concat_proto, Identity_proto
from searching.cnn.proxy_estimator import CIFARProxyEstimator

import os, sys
import pickle

import matplotlib.pyplot as plt

from math import log10

from api.backend import G

from tqdm import tqdm

def optimize_mask(mask):
    """
    Remove redundant edges in the original mask.
    :param mask:
    :return:
    """
    mask_shape = np.shape(mask)
    print(mask_shape)
    nodes = mask_shape[0]

    new_mask = np.zeros_like(mask)

    nodes_queue = [0]
    for d1 in range(nodes):
        if not d1 in nodes_queue:
            continue
        for d2 in range(d1 + 1, nodes):
            if mask[d1][d2] == 1:
                new_mask[d1][d2] = 1
                nodes_queue.append(d2)
    return new_mask


def convert_to_upper_triangle(mask):
    n = mask.shape[0]
    feats = []
    for i in range(n):
        for j in range(i+1,n):
            feats.append(mask[i][j])
    return feats

def stat_num_concats(mask):
    """
    :param mask: Must be optimized.
    :return:
    """
    mask_shape = np.shape(mask)
    nodes = mask_shape[0]

    # Usually final one should count.
    num_concats = 1
    for d1 in range(nodes):
        degree = 0
        for d2 in range(d1):
            if mask[d2][d1] == 1:
                degree += 1
        if degree > 1:
            num_concats += 1
    return num_concats

registered_protocols = {
    'v3': ops_gen.ops_gen_v3,
}

class MetaGraph:
    """
    GRAM metagraph v2. Decreasing the edges from a pre-defined architecture to test the performance.
    """
    def __init__(self, nodes=10,
                 depth=3, width=1,
                 p=1.0,
                 logdir="temp",
                 model_root="./v3_graph/demo",
                 solvertxt="./solver/cifar10-solver.prototxt",
                 protocol='v3'):
        self.nodes = nodes
        self.depth = depth
        self.width = width
        self.edges = 0
        raw_mask = np.zeros(shape=[nodes, nodes])
        nodes_queue = [0]
        for i in range(nodes):
            if not i in nodes_queue:
                continue
            for j in range(i+1, nodes):
                if np.random.rand() <= p:
                    raw_mask[i][j] = 1
                    nodes_queue.append(j)
                    self.edges += 1
        self.mask = raw_mask.copy()
        if protocol in registered_protocols.keys():
            self.protocol = registered_protocols[protocol]()
        else:
            raise NotImplementedError
        # Create ops def
        self.model_root = model_root
        if not os.path.exists(self.model_root):
            os.makedirs(self.model_root)
        self.ops_def = self.protocol.seed_group_ops(self.nodes)
        self.raw_mask = self.mask.copy()
        self.edges = util.inspect_num_edges(self.mask)
        self.max_edges = self.edges
        self.iteration = 0
        self.ifstream = 'input'
        self.ofstream = None
        self.logdir = model_root
        self.solvertxt = solvertxt
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        # Record the move sequence of edges
        self.move_sequence = np.zeros_like(self.mask)
        self.move_count = 0

        # Add logging
        self.mean_score_list = []
        self.edges_left_list = []
        self.mean_acc_list = []
        self.mean_latency_list = []
        self.best_acc_list = []
        self.best_perf_list = []

        self.perf_log_path = os.path.join(self.model_root, 'perf_log.png')
        self.acc_log_path = os.path.join(self.model_root, 'mean_acc.png')
        self.latency_log_path = os.path.join(self.model_root, 'mean_latency.png')
        self.best_perf_log_path = os.path.join(model_root, 'best_perf_log.png')
        self.best_acc_log_path = os.path.join(model_root, 'best_acc.png')

        self.mean_acc_txt = os.path.join(self.model_root, 'mean_acc.log')
        self.mean_latency_txt = os.path.join(self.model_root, 'mean_latency.log')
        self.perf_txt = os.path.join(self.model_root, 'perf.log')
        self.best_acc_txt = os.path.join(model_root, 'best_acc.log')
        self.best_perf_txt = os.path.join(model_root, 'best_perf.log')

        # Add internal edge queue
        self._internal_edge_queue = self.fetch_queue()


    def fetch_queue(self):
        """
        Fetch the latest edge queue
        :return:
        """
        self.mask = self.raw_mask.copy()
        # create edge list
        edge_list = []
        nodes_queue = [0]
        for i in range(self.nodes):
            if not i in nodes_queue:
                continue
            for j in range(i+1, self.nodes):
                if self.mask[i][j]:
                    nodes_queue.append(j)
                    edge_list.append([i, j])
        self._internal_edge_queue = edge_list
        np.random.shuffle(self._internal_edge_queue)
        return self._internal_edge_queue

    def random_remove(self):
        """
        Randomly remove one edge
        :return:
        """
        edge_to_remove = self._internal_edge_queue[0]
        self.mask[edge_to_remove[0]][edge_to_remove[1]] = 0
        self._internal_edge_queue = self._internal_edge_queue[1:]
        np.random.shuffle(self._internal_edge_queue)
        print("Removing %s ..." %edge_to_remove)
        return edge_to_remove

    def permanently_remove(self, i, j):
        """
        Permanently remove edge i, j
        :param i:
        :param j:
        :return:
        """
        self.raw_mask[i][j] = 0
        self.edges -= 1
        self.mask = self.raw_mask.copy()

    def add_to_proto(self, protoWriter, replicate=None, scale=None, pool=None,
                     activation='relu', use_bias=True, use_residual=True,
                     bottleneck_factor=1):
        """
        Add masked network to ProtoWriter instance, thus it can serve different applications.
        :param protoWriter: Protowriter instance.
        :param replicate: Arrangement of searched cell in each stage. Can be either 'int' or 'tuple'.
        If an integer is specified, the searched architecture will have the same number of cells in each stage.
        :param scale: Scale for each search cell. Can be either 'int' or 'tuple'.
        If an integer is entered, the searched architecture will have cells of the same width in each stage.
        :param pool: Pooling size for each stage. Can be either 'int' or 'tuple', and '1' means no MaxPooling.
        If an integer is entered, the searched architecture will have same maxpooling pool size in each stage.
        :param activation: non-linear activation function for each layer, except for the 'linear bottleneck'.
        Can be one of ['relu', 'relu6', 'swish', leaky', 'linear'].
        :param use_bias: Whether to use bias in each layer. Can be 'True' or 'False'.
        :param use_residual: Whether to formulate an additional residual connection in each cell.
        :param bottleneck_factor: Specifies the bottleneck factor in residual blocks.
        Sometimes bottlenecked residual blocks can be used. Can be greater than 1 (bottleneck) or
        lower than 1 (inverted bottleneck).
        :return: None
        """
        if replicate is None:
            _replicate = np.ones(self.depth, dtype=np.int)
        elif isinstance(replicate, int):
            _replicate = np.zeros(self.depth, dtype=np.int) + replicate
        else:
            _replicate = replicate

        if scale is None:
            _scale = np.ones(self.depth, dtype=np.float)
        elif isinstance(scale, float):
            _scale = np.zeros(self.depth, dtype=np.float) + scale
        else:
            _scale = scale

        if pool is None:
            _pool = np.ones(self.depth, dtype=np.int) * 2
        elif isinstance(pool, int):
            _pool = np.zeros(self.depth, dtype=np.int) + pool
        else:
            _pool = pool

        if self.depth != len(_replicate):
            print("Warning: Setting depth to %d" %len(_replicate))
            self.set_depth(len(_replicate))

        ifstream = []
        for i in range(self.width):
            ifstream.append(self.ifstream)
        for i in range(self.depth):
            for j in range(self.width):
                replicas_ofstream = []
                for replicas in range(_replicate[i]):
                    if replicas == 0:
                        _strides = _pool[i]
                    else:
                        _strides = 1
                    scope = "DAG_%d_%d/replica_%d" %(i+1, j+1, replicas)
                    _ifstream = ifstream[j]
                    proto, ofstream = util.convert_mask_to_proto(self.mask,
                                                                 self.ops_def,
                                                                 _ifstream,
                                                                 scale=_scale[i],
                                                                 scope=scope,
                                                                 strides=_strides,
                                                                 use_bias=use_bias,
                                                                 activation=activation,
                                                                 use_residual=use_residual,
                                                                 bottleneck_factor=bottleneck_factor)
                    protoWriter.add(proto)
                    ifstream[j] = ofstream
                    replicas_ofstream.append(ofstream)
            # concat all layers from different width position
            concat_name = "concat_pool_%d" %(i+1)
            if len(ifstream) == 1:
                proto = Identity_proto(name=concat_name,
                                       input=ifstream[0])
            else:
                proto = Concat_proto(name=concat_name,
                                     input=ifstream)
            protoWriter.add(proto)

            self.ofstream = concat_name

            for k in range(self.width):
                ifstream[k] = self.ofstream
        pass

    def metric_fn(self, accuracy,
                  flops):
        """
        Object function for calculating the search objective. Take 'accuracy' and 'flops' as input.
        :param accuracy: Computed accuracy from the validation proxy dataset.
        :param flops: Computed FLOPS (MACS) of the network.
        :return: Computed metric.
        """
        return accuracy - (log10(flops)-6) * 0.1

    def auto_shrink(self, max_steps_to_action=10, keep_fraction=0.5,
                    drop_k_each_iteration=1,
                    scale=None,
                    replicate=None,
                    pool=None,
                    use_bias=False):
        """
        Auto remove edges within a number of actions
        :param max_steps_to_action: Max step to force an edge removal
        :param keep_fraction: Fraction of edges to keep
        :return:
        """
        self.max_edges = util.inspect_num_edges(self.mask)
        print("Initial Graph has %d Edges!" %self.max_edges)
        edges_threshold = round(self.max_edges * keep_fraction)
        while self.edges > edges_threshold:
            edges_to_be_removed_collection = []
            val_collection = []
            acc_collection = []
            latency_collection = []
            self._internal_edge_queue = self.fetch_queue()
            eval_steps = min(max_steps_to_action, len(self._internal_edge_queue))
            for i in range(eval_steps):
                self.mask = self.raw_mask.copy()
                edge_to_remove = self.random_remove()
                print(self.mask.sum())
                writer = ProtoWriter("v3/runtime-%d" %self.iteration)
                writer.add_header(task='cifar10')
                self.ifstream = writer.ifstream
                self.add_to_proto(writer,
                                  scale=scale,
                                  replicate=replicate,
                                  pool=pool,
                                  use_bias=use_bias)
                writer.finalized(task='cifar10', outstream_name=self.ofstream)
                writer.set_global_regularization(1e-5)
                prototxt = os.path.join(self.logdir, "runtime-%d.prototxt" %self.iteration)
                with open(prototxt, 'w') as fp:
                    writer.dump(fp)

                # Use estimator to evaluate performance
                estimator = CIFARProxyEstimator(prototxt=prototxt,
                                                solvertxt=self.solvertxt,
                                                verbose=1)
                accuracy, flops = estimator.trainval_on_proxy(epochs=20)
                val = self.metric_fn(accuracy, flops)
                print(val)
                print(edge_to_remove)
                print("FLOPS: %.2f M" %(flops / 1e6))
                edges_to_be_removed_collection.append(edge_to_remove)
                val_collection.append(val)
                acc_collection.append(accuracy)
                latency_collection.append(flops)
                self.iteration += 1
                print("Iteration %d" %self.iteration)
            # Remove the worst edge
            mean_score = np.mean(val_collection)
            mean_acc = np.mean(acc_collection)
            mean_latency = np.mean(latency_collection)
            best_acc = np.max(acc_collection)
            best_score = np.max(val_collection)

            print("Mean score: %.4f" % mean_score)
            sorted_args = np.argsort(val_collection)
            for num in range(drop_k_each_iteration):
                print(edges_to_be_removed_collection)
                i, j = edges_to_be_removed_collection[sorted_args[-1-num]]
                print("Dropping (%d, %d)..." %(i, j))
                self.permanently_remove(i, j)
                self.move_sequence[i][j] = self.move_count
                self.move_count += 1

            # optimze to remove redundant edges
            self.raw_mask = optimize_mask(self.raw_mask)

            self.edges = util.inspect_num_edges(self.mask)
            print("%d Edges remaining..." %self.edges)

            self.edges_left_list.append(self.edges)

            self.mean_score_list.append(mean_score)
            self.mean_acc_list.append(mean_acc)
            self.mean_latency_list.append(mean_latency / 1e6)
            self.best_acc_list.append(best_acc)
            self.best_perf_list.append(best_score)

            edges_remain_ratio = float(self.edges / self.max_edges)
            tf.logging.info(".2f % edges remained. Saving models...")
            pb_name = os.path.join(self.model_root, "graph-%.2f" % edges_remain_ratio)
            with open(pb_name, 'wb') as fp:
                pickle.dump(self, fp)

            # Logging mean score vs edges
            plt.figure()
            plt.plot(self.edges_left_list, self.mean_score_list)
            plt.xlabel("Remaining edges")
            plt.ylabel("Mean Score")
            plt.savefig(self.perf_log_path)
            plt.close()

            # Logging acc only
            plt.figure()
            plt.plot(self.edges_left_list, self.mean_acc_list)
            plt.xlabel("Remaining edges")
            plt.ylabel("Mean Accuracy (%)")
            plt.savefig(self.acc_log_path)
            plt.close()

            # Loading latency only
            plt.figure()
            plt.plot(self.edges_left_list, self.mean_latency_list)
            plt.xlabel("Remaining edges")
            plt.ylabel("Mean Latency (M)")
            plt.savefig(self.latency_log_path)
            plt.close()

            # Logging best score only
            plt.figure()
            plt.plot(self.edges_left_list, self.best_perf_list)
            plt.xlabel("Remaining edges")
            plt.ylabel("Best score (M)")
            plt.savefig(self.best_perf_log_path)
            plt.close()

            # Logging best acc only
            plt.figure()
            plt.plot(self.edges_left_list, self.best_acc_list)
            plt.xlabel("Remaining edges")
            plt.ylabel("Best acc (M)")
            plt.savefig(self.best_acc_log_path)
            plt.close()

        pass

        # Dump txt file
        self._dump_to_txt(self.edges_left_list, self.mean_score_list, self.perf_txt)
        self._dump_to_txt(self.edges_left_list, self.mean_acc_list, self.mean_acc_txt)
        self._dump_to_txt(self.edges_left_list, self.mean_latency_list, self.mean_latency_txt)
        self._dump_to_txt(self.edges_left_list, self.best_acc_list, self.best_acc_txt)
        self._dump_to_txt(self.edges_left_list, self.best_perf_list, self.best_perf_txt)

        tf.logging.info("Final Graph...")
        writer = ProtoWriter("v3/runtime-%d" % self.iteration)
        writer.add_header(task='cifar10')
        self.ifstream = writer.ifstream
        self.add_to_proto(writer)
        writer.finalized(task='cifar10', outstream_name=self.ofstream)
        writer.set_global_regularization(1e-5)
        prototxt = os.path.join(self.logdir, "runtime-%d.prototxt" % self.iteration)
        with open(prototxt, 'w') as fp:
            writer.dump(fp)
        estimator = CIFARProxyEstimator(prototxt=prototxt,
                                        solvertxt=self.solvertxt,
                                        verbose=2)
        accuracy, flops = estimator.trainval_on_proxy()
        print("Accuracy: %.4f" %(accuracy))
        print("Flops: %.2f M" %(flops / 1e6))



    def create_eval_graph(self, replicate=None, scale=None,
                          model_root="v3/eval",
                          channel_scale=1.0,
                          pool=None,
                          activation='relu',
                          use_bias=True,
                          use_residual=False,
                          bottleneck_factor=4,
                          task='cifar10'):
        """
        Create a evaluation prototxt and dump to text file.
        :param replicate: Arrangement of searched cell in each stage. Can be either 'int' or 'tuple'.
        If an integer is specified, the searched architecture will have the same number of cells in each stage.
        :param scale: Scale for each search cell. Can be either 'int' or 'tuple'.
        If an integer is entered, the searched architecture will have cells of the same width in each stage.
        :param model_root:
        :param channel_scale:
        :param pool: Pooling size for each stage. Can be either 'int' or 'tuple', and '1' means no MaxPooling.
        If an integer is entered, the searched architecture will have same maxpooling pool size in each stage.
        :param activation: non-linear activation function for each layer, except for the 'linear bottleneck'.
        Can be one of ['relu', 'relu6', 'swish', leaky', 'linear'].
        :param use_bias: Whether to use bias in each layer. Can be 'True' or 'False'.
        :param use_residual: Whether to formulate an additional residual connection in each cell.
        :param bottleneck_factor: Specifies the bottleneck factor in residual blocks.
        Sometimes bottlenecked residual blocks can be used. Can be greater than 1 (bottleneck) or
        lower than 1 (inverted bottleneck).
        :param task: Target task. Please check 'api/proto/lazy_loader.py' for details.
        :return: proto writer instance
        """
        if replicate is None:
            _replicate = np.ones(self.depth)
        else:
            _replicate = replicate
        if scale is None:
            _scale = np.ones(self.depth)
        else:
            _scale = scale
        _scale = np.asarray(_scale) * channel_scale
        writer_path = os.path.join(model_root, "channel-scale-%.2f" %channel_scale, task)
        if use_residual:
            writer_path = os.path.join(writer_path, "res-%d" %bottleneck_factor)
        else:
            writer_path = os.path.join(writer_path, "plain")
        writer = ProtoWriter(writer_path)
        writer.add_header(task=task)
        writer.scaling(channel_scale)
        self.ifstream = writer.ifstream
        self.add_to_proto(writer, replicate=_replicate, scale=_scale, pool=pool,
                          activation=activation, use_bias=use_bias, use_residual=use_residual,
                          bottleneck_factor=bottleneck_factor)
        writer.finalized(task=task, outstream_name=self.ofstream)
        return writer

    def set_width(self, new_width):
        """
        Set depth and width for task transferbility
        :param new_width:
        :return:
        """
        self.width = new_width

    def set_depth(self, new_depth):
        self.depth = new_depth

    def _dump_to_txt(self, x, y, fname):
        """

        :param x: domain variables
        :param y: codomain variables
        :param fname:
        :return:
        """
        with open(fname, 'w') as fp:
            l = len(x)
            for i in range(l):
                fp.write("%s:\t%s\n" %(x[i], y[i]))

