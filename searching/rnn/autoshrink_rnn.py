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

import numpy as np
import os, sys
import pickle
import matplotlib.pyplot as plt
from math import log10

cwd = os.getcwd()
sys.path.append(cwd)

import searching.rnn.utils as utils
from searching.rnn.model import Estimator
import torch.nn as nn
import torch

class AutoShrinkRNN:
    def __init__(self, args, ntoken,
                 logging, emb, nodes=10, p=1.0,
                 ):
        self.continue_train = args.continue_train
        self.logging = logging
        self.save = args.save
        self.epochs = args.epochs
        self.nodes = nodes
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
        self.mask = raw_mask
        # Create ops def
        self.ops_def = utils.random_choose_ops(self.nodes)
        self.edges = utils.inspect_num_edges(self.mask)
        self.max_edges = self.edges
        self.iteration = 0
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

        self.perf_log_path = os.path.join(self.save, 'perf_log.png')
        self.acc_log_path = os.path.join(self.save, 'mean_acc.png')
        self.latency_log_path = os.path.join(self.save, 'mean_latency.png')
        self.best_perf_log_path = os.path.join(self.save, 'best_perf_log.png')
        self.best_acc_log_path =  os.path.join(self.save, 'best_acc.png')

        # Add internal edge queue & fetch edge list
        self._internal_edge_queue = self.fetch_queue()


    def fetch_queue(self):
        """
        Fetch the latest edge queue
        :return:
        """
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


    # Randomly remove one edge, and reshuffle the edge queue.
    def random_remove(self):
        edge_to_remove = self._internal_edge_queue[0]
        self.mask[edge_to_remove[0]][edge_to_remove[1]] = 0
        self._internal_edge_queue = self._internal_edge_queue[1:]
        np.random.shuffle(self._internal_edge_queue)
        print("Removing %s ..." %edge_to_remove)
        return edge_to_remove

    def recover_edge(self, edge):
        self.mask[edge[0]][edge[1]] = 1
        pass

    def permanently_remove(self, i, j):
        """
        Permanently remove edge i, j
        :param i:
        :param j:
        :return:
        """
        self.mask[i][j] = 0
        self.edges -= 1

    # 由acc和flop获得score
    def metric_fn(self, ppl,
                  flops):
        """
        Write a metric function.
        :param ppl:
        :param flops:
        :return:
        """
        return (200 - ppl) * 0.25 - (log10(flops)-6) 

    def auto_shrink(self, estimator, train_data, val_data, max_steps_to_action=10, keep_fraction=0.5,
                    drop_k_each_iteration=1,
                    save_n_iterations=10
                    ):
        """
        Auto remove edges within a number of actions
        :param max_steps_to_action: Max step to force an edge removal
        :param keep_fraction: Fraction of edges to keep
        :return:
        """
        if not self.continue_train:
            self.max_edges = utils.inspect_num_edges(self.mask)
            print("Initial Graph has %d Edges!" %self.max_edges)
        else:
            print("Graph has %d Edges now!" %self.max_edges)
        edges_threshold = round(self.max_edges * keep_fraction)
        while self.edges > edges_threshold:
            edges_to_be_removed_collection = []
            val_collection = []
            acc_collection = []
            latency_collection = []
            self._internal_edge_queue = self.fetch_queue()
            eval_steps = min(max_steps_to_action, len(self._internal_edge_queue))
            
            for i in range(eval_steps):
                edge_to_remove = self.random_remove()
                # 如果过拟合、或者训练效果差、或者连续若干epoch不提升 会提前终止
                ppl, flops =estimator.train_and_eval(self.mask, self.ops_def,
                                                                train_data,val_data,
                                                                self.epochs, self.logging)
                self.recover_edge(edge_to_remove)
                val = self.metric_fn(ppl, flops)
                print(val)
                print(edge_to_remove)
                print("FLOPS: %.2f M" %(flops / 1e6))
                edges_to_be_removed_collection.append(edge_to_remove)
                val_collection.append(val)
                acc_collection.append(ppl)
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
                self.move_sequence[i][j] = self.move_count + 1
                self.move_count += 1
            # optimze to remove redundant edges
            self.mask = utils.optimize_mask(self.mask)


            self.edges = utils.inspect_num_edges(self.mask)
            print("%d Edges remaining..." %self.edges)

            self.edges_left_list.append(self.edges)

            self.mean_score_list.append(mean_score)
            self.mean_acc_list.append(mean_acc)
            self.mean_latency_list.append(mean_latency / 1e6)
            self.best_acc_list.append(best_acc)
            self.best_perf_list.append(best_score)

            edges_remain_ratio = float(self.edges / self.max_edges)
            self.logging.info(".2f % edges remained. Saving models...")
            pb_name = os.path.join(self.save, "graph-%.2f" % edges_remain_ratio)
            with open(pb_name, 'wb') as fp:
                torch.save(estimator.rnn, os.path.join(self.save, 'rnn.pt'))
                #print(self.__dir__())
                #for s in self.__dir__(): eval('print(self.'+s+',type(self.'+s+'))')
                tmp = self.logging
                self.logging = None
                pickle.dump(self, fp)
                self.logging = tmp
          
            # Logging mean score vs edges
            #self.draw(self.mean_score_list, "Mean Score", self.perf_log_path)
            #self.draw(self.mean_acc_list, "Mean Accuracy (%)", self.acc_log_path)
            #self.draw(self.mean_latency_list, "Mean Latency (M)", self.latency_log_path)
            #self.draw(self.best_perf_list, "Best Score (M)", self.best_perf_log_path)
            #self.draw(self.best_acc_list, "Best Accuracy (M)", self.best_acc_log_path)
        pass

        # Dump txt file
        #self._dump_to_txt(self.edges_left_list, self.mean_score_list, self.perf_txt)
        #self._dump_to_txt(self.edges_left_list, self.mean_acc_list, self.mean_acc_txt)
        #self._dump_to_txt(self.edges_left_list, self.mean_latency_list, self.mean_latency_txt)
        #self._dump_to_txt(self.edges_left_list, self.best_acc_list, self.best_acc_txt)
        #self._dump_to_txt(self.edges_left_list, self.best_perf_list, self.best_perf_txt)

        self.logging.info("Final Graph...")
        '''
        writer = ProtoWriter("v3/runtime-%d" % self.iteration)
        writer.add_header(task='cifar10')
        self.ifstream = writer.ifstream
        self.add_to_proto(writer)
        writer.finalized(task='cifar10', outstream_name=self.ofstream)
        writer.set_global_regularization(1e-5)
        prototxt = os.path.join(self.logdir, "runtime-%d.prototxt" % self.iteration)
        with open(prototxt, 'w') as fp:
            writer.dump(fp)
        '''
        accuracy, flops = estimator.train_and_eval(self.mask, self.ops_def,
                                                        train_data,val_data,
                                                        self.epochs, self.logging)
        print("Accuracy: %.4f" %(accuracy))
        print("Flops: %.2f M" %(flops / 1e6))

    def draw(self, ydata, yname, save_path):
        plt.figure()
        plt.plot(self.edges_left_list, ydata)
        plt.xlabel("Remaining edges")
        plt.ylabel(yname)
        plt.savefig(save_path)
        plt.close()

    def auto_shrink_with_compression(self, max_steps_to_action=10, keep_fraction=0.5,
                    drop_k_each_iteration=1,
                    save_n_iterations=10,
                    scale=None,
                    replicate=None,
                    pool=None):
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
                edge_to_remove = self.random_remove()
                writer = ProtoWriter("v3/runtime-%d" %self.iteration)
                writer.add_header(task='cifar10')
                self.ifstream = writer.ifstream
                self.add_to_proto(writer,
                                  scale=scale,
                                  replicate=replicate,
                                  pool=pool)
                writer.finalized(task='cifar10', outstream_name=self.ofstream)
                writer.set_global_regularization(1e-5)
                prototxt = os.path.join(self.logdir, "runtime-%d.prototxt" %self.iteration)
                with open(prototxt, 'w') as fp:
                    writer.dump(fp)

                # Use estimator to evaluate performance
                estimator = CIFARCompressedProxyEstimator(prototxt=prototxt,
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

            # optimze to remove redundant edges
            self.raw_mask = optimize_mask(self.raw_mask)

            print("Mean score: %.4f" % mean_score)
            sorted_args = np.argsort(val_collection)
            for num in range(drop_k_each_iteration):
                print(edges_to_be_removed_collection)
                i, j = edges_to_be_removed_collection[sorted_args[-1-num]]
                print("Dropping (%d, %d)..." %(i, j))
                self.permanently_remove(i, j)
                self.move_sequence[i][j] = self.move_count
                self.move_count += 1

            self.edges = utils.inspect_num_edges(self.mask)
            print("%d Edges remaining..." %self.edges)

            self.edges_left_list.append(self.edges)

            self.mean_score_list.append(mean_score)
            self.mean_acc_list.append(mean_acc)
            self.mean_latency_list.append(mean_latency / 1e6)
            self.best_acc_list.append(best_acc)
            self.best_perf_list.append(best_score)

            edges_remain_ratio = float(self.edges / self.max_edges)
            self.logging.info(".2f % edges remained. Saving models...")
            pb_name = os.path.join(self.save, "graph-%.2f" % edges_remain_ratio)
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
        estimator = CIFARCompressedProxyEstimator(prototxt=prototxt,
                                        solvertxt=self.solvertxt,
                                        verbose=2)
        accuracy, flops = estimator.trainval_on_proxy()
        print("Accuracy: %.4f" %(accuracy))
        print("Flops: %.2f M" %(flops / 1e6))

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



