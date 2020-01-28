import api.tf_lib as tflib
import tensorflow as tf

from api.network.Parser import *
from tensorflow.python.platform import gfile
from api.backend import G
import os
from api.network.gpu_devices import *

from tensorflow.python.tools import freeze_graph

from api.network.load_checkpoint import get_vars_to_restore, get_tensors_by_name


# For visualization
l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))

ps_device = "gpu:0"


class Model:
    def __init__(self, deploy=None, solver=None, ngpus=1, nthreads=44,
                 data_train_op=None, data_val_op=None, verbose=False,
                 name_scope='feature_extractor',
                 batch_size=None, create_quantize_op=False,
                 display_interval=1000,
                 inference_only=False,
                 inference_type=tf.float32,
                 train_only=False,
                 compile=True):
        """
        High-level API using tensorflow backend. Transferring prototxts to tensorflow model and train with solver.
        :param deploy: Model deployment file.
        :param solver: Solver used to train the model.
        :param ngpus: Number of GPUs to use.
        :param nthreads: Maximum number of threads to use.
        :param data_train_op: Training Data input tensors. If 'data_train_op' and 'data_val_op' are all specified,
        it will override the default 'feed_dict' mechanism and no data is required
        in the 'train_on_batch' and 'test_on_batch' method. This will lead to performance gain.
        :param data_val_op: Validation Data input tensors. If 'data_train_op' and 'data_val_op' are all specified,
        it will override the default 'feed_dict' mechanism and no data is required
        in the 'train_on_batch' and 'test_on_batch' method. This will lead to performance gain.
        :param batch_size: Batch size specified. If you do not use a tfrecord, leave it to None. (Experimental feature.)
        """
        if G.BACKEND == "tensorflow":
            self.ngpus = ngpus
            self.nthreads = nthreads
            self.layerinfos = {}
            self.hyper_feed_dict = {}
            self.hyper_param_dict = {}
            self.layers = []
            self.layer_gpu_id = []
            self.metrics = []
            self.loss = None
            self.ngpus_loss_p = None
            self.ngpus_loss = []
            self.reg_loss = None
            self.ngpus_reg_losses = []
            self.input_tensors = []
            self.metric_tensors = []
            self.num_trainable_parameters = 0
            self.num_non_trainable_parameters = 0
            self.shared_trainable_parameters = 0
            self.mem_cost = 0
            self.peak_mem = 0
            self.verbose = verbose
            self.name_scope = name_scope
            self.batch_size = batch_size
            self.inference_only = inference_only
            self.inference_type = inference_type
            self.train_only = train_only

            self.global_epochs = 0

            if inference_only:
                self.is_training = tf.convert_to_tensor(False)
            else:
                if train_only:
                    self.is_training = tf.convert_to_tensor(True)
                else:
                    self.is_training = tf.placeholder(shape=(), dtype=tf.bool)
            self.input_collections = []
            self.ngpu_input_tensors = {}

            self.model_root = None
            self.save_path = None
            self.log_path = None
            self.pretrain_path = None
            self.drop_last_nvars = 0

            self.train_dropout_dict = {}
            self.test_dropout_dict = {}

            self.initial_feed_dict = {}

            self.output = []
            self.train_ops = []
            self.flops = 0
            self.raw_graph = None

            self.summary_writer = None
            self.summary_op = None
            self.summary_display = display_interval

            self.next_step_op = None

            self.assign_op = None
            self.assign_tensors = None

            self.fp_weights_copy = None

            self.converted_to_quantized = False

            if data_train_op is not None and data_val_op is not None:
                self.next_step_op = tf.cond(self.is_training, lambda: data_train_op, lambda: data_val_op)
            else:
                assert self.batch_size is None, \
                    "batch_size must not be specified if input is not coming from tfrecord."

            self._load_model(deploy)
            self._load_solver(solver)
            self.global_step = tf.placeholder(shape=(), dtype=tf.float32)
            self.global_steps = 0
            self.feed_dict = {}
            if create_quantize_op:
                print("Prepare weight assign ops for quantization")
                self.create_assign_weights_op()
            if compile:
                self.compile()
        else:
            raise NotImplementedError

    def create_assign_weights_op(self):
        assign_ops = []
        self.assign_tensors = []
        for var in tf.trainable_variables():
            assign_w = tf.placeholder(tf.float32, var.shape)
            assign_op = tf.assign(var, assign_w)
            assign_ops.append(assign_op)
            self.assign_tensors.append(assign_w)

        self.assign_op = tf.group(assign_ops)

    def compile(self):
        """
        Compile the model.
        :return:
        """
        self.ngpus_loss_p = tf.reduce_mean(self.ngpus_loss)

        tf.summary.scalar("loss/weight_loss", self.ngpus_reg_losses)
        tf.summary.scalar("loss/cross_entropy_loss", self.ngpus_loss_p)
        if self.pretrain_path is None:
            latest_checkpoint = None
        else:
            latest_checkpoint = tf.train.latest_checkpoint(self.pretrain_path)

        lr = ModelAssign(self.hyper_param_dict, 'lr', 1e-4)
        self.lr = tf.placeholder(shape=(), dtype=tf.float32, name="learning_rate")
        self.global_epoch = tf.placeholder(shape=(), dtype=tf.int32, name="global_epoch")

        if not self.inference_only:
            tf.logging.info("Create Loss and Backward ops...")
            if self.inference_type == tf.uint8:
                tf.logging.info("Adding quantization ops...")
                tf.contrib.quantize.create_training_graph(self.sess.graph)
                print("Done")

            self.optimizer_name = ModelAssign(self.hyper_param_dict, 'optimizer', 'adam')

            # Fetch learning rate decay method
            self.decay_method = tflib.Decay[ModelAssign(self.hyper_param_dict, 'decay_method', 'no_decay')]
            self.decay = ModelAssign(self.hyper_param_dict, 'decay', 0.0)
            self.decay_steps = ModelAssign(self.hyper_param_dict, 'decay_steps', 350)

            self.global_epochs = ModelAssign(self.hyper_param_dict, 'start_steps', 0)
            if self.global_epochs != 0:
                print("Starting from ...Epoch %d" % self.global_epochs)

            self.scheduled_lr = self.decay_method(lr=self.lr, global_step=self.global_epoch,
                                                  decay_steps=self.decay_steps, decay_rate=self.decay)
            tf.summary.scalar("Configuration/Learning_Rate", self.scheduled_lr)

            lr = ModelAssign(self.hyper_param_dict, 'lr', 1e-4)
            self.hyper_feed_dict.update({self.lr: lr * self.ngpus})

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            #ema = tf.train.ExponentialMovingAverage(decay=0.9999)
            #var_list = tf.get_collection("trainable_variables")

            opt = tflib.Optimizer[self.optimizer_name](self.scheduled_lr)
            # print("Enabling the mixed precision mode...")
            # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
            opt = tf.contrib.opt.MovingAverageOptimizer(opt, 0.999, num_updates=self.global_step)
            trainable_vars = tf.trainable_variables()

            if self.ngpus != 1:
                grad_tower = []
                for i in range(self.ngpus):
                    with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device=ps_device)):
                        if i == 0:
                            total_loss = self.ngpus_loss[i] + self.ngpus_reg_losses * self.ngpus
                        else:
                            total_loss = self.ngpus_loss[i]
                        grads_and_vars = opt.compute_gradients(total_loss, colocate_gradients_with_ops=True, var_list=trainable_vars)
                        grad_tower.append(grads_and_vars)
                with tf.device(ps_device):
                    grads_and_vars = average_gradients(grad_tower)
            else:
                total_loss = self.ngpus_loss[0] + self.ngpus_reg_losses
                grads_and_vars = opt.compute_gradients(total_loss, colocate_gradients_with_ops=True, var_list=trainable_vars)

            with tf.control_dependencies(update_ops):
                # Gradient clipping
                # grads_and_vars = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads_and_vars]
                self.train_ops = opt.apply_gradients(grads_and_vars)

            # Finally, open session
            if G.USE_ALL_GPU == 1:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            else:
                gpu_options = tf.GPUOptions(allow_growth=True)
            session_conf = tf.ConfigProto(  intra_op_parallelism_threads=self.nthreads,
                                            inter_op_parallelism_threads=self.nthreads,
                                            gpu_options=gpu_options,
                                            log_device_placement=False,
                                            allow_soft_placement=True)
            self.sess = tf.InteractiveSession(config=session_conf)

            # Initialize summary op
            self.summary_writer = tf.summary.FileWriter(self.log_path, self.sess.graph)
            self.summary_op = tf.summary.merge_all()
        else:
            # Finally, open session
            if G.USE_ALL_GPU == 1:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            else:
                gpu_options = tf.GPUOptions(allow_growth=True)
            session_conf = tf.ConfigProto(  intra_op_parallelism_threads=self.nthreads,
                                            inter_op_parallelism_threads=self.nthreads,
                                            gpu_options=gpu_options,
                                            log_device_placement=False,
                                            allow_soft_placement=True)
            self.sess = tf.InteractiveSession(config=session_conf)

            # Initialize summary op
            self.summary_writer = tf.summary.FileWriter(self.log_path, self.sess.graph)
            self.summary_op = tf.summary.merge_all()

            if self.inference_type == tf.uint8:
                tf.logging.info("Creating eval graph...")
                tf.contrib.quantize.create_eval_graph(self.sess.graph)
                print("Done")

        tf.logging.info("Pretrained path %s" % self.pretrain_path)


        self.sess.run(tf.global_variables_initializer(),
                      feed_dict={self.lr: lr})
        if latest_checkpoint is not None:
            vars_to_restore = get_vars_to_restore(latest_checkpoint)
            vars_to_restore = get_tensors_by_name(vars_to_restore)
        else:
            tf.logging.info("Checkpoint is None.")
            vars_to_restore = []

        if self.drop_last_nvars != 0 and vars_to_restore != []:
            tf.logging.info("Drop last %d variables..." %self.drop_last_nvars)
            vars_to_restore = vars_to_restore[:-self.drop_last_nvars]
        else:
            tf.logging.info("Restore all variables...")

        # Define saver and restore variables
        if vars_to_restore != []:
            try:
                self.saver = opt.swapping_saver()
            except Exception as e:
                print(e)
                print("Swapping saver is not available. Use default saver instead.")
                self.saver = tf.train.Saver(var_list=vars_to_restore, max_to_keep=5)
        else:
            try:
                self.saver = opt.swapping_saver()
            except Exception as e:
                print(e)
                print("Swapping saver is not available. Use default saver instead.")
                self.saver = tf.train.Saver(max_to_keep=5)

        if self.pretrain_path is not None:
            tf.logging.info(latest_checkpoint)
            try:
                self.saver.restore(self.sess, latest_checkpoint)
            except Exception as e:
                tf.logging.warning("Loading pretrained weights failed! Use the default initializer.")
                print(e)

        if not self.inference_only:
            try:
                self.saver = opt.swapping_saver()
            except Exception as e:
                print(e)
                print("Swapping saver is not available. Use default saver instead.")
                self.saver = tf.train.Saver(var_list=vars_to_restore, max_to_keep=5)
                #self.sess.graph.finalize()
        else:
            print("Use inference only saver...")
            self.saver = tf.train.Saver(var_list=vars_to_restore, max_to_keep=5)

    def summary(self):
        """
        Print the model summary.
        :return: None
        """
        print("--------------------------------------------------------------------------------------------"*2)
        print("|----------Layer---------------|----------Input-->Output------------|------Trainable Params------|----"
              "--Non-trainable params------|----Shared params----|------Skip From-----|")
        for i in range(len(self.layers)):
            if self.layer_gpu_id[i] == "general" or self.layer_gpu_id[i] == "gpu_0":
                self.layers[i].summary()
        print("Total trainable parameters: %d" %self.num_trainable_parameters)
        print("Total non-trainable parameters: %d" %self.num_non_trainable_parameters)
        print("Total shared-weight trainable parameters: %d" %self.shared_trainable_parameters)
        print("Total MACs: %.2f M" %(self.flops / 1e6))
        print("Total Memory Cost: %.2f M" %(self.mem_cost / 1e6))
        print("Peak Memory Cost: %.2f K" %(self.peak_mem / 1e3))
        print("--------------------------------------------------------------------------------------------"*2)
        #print(tf.trainable_variables())

    def train(self):
        """
        Enable the training phase
        :return:
        """
        self.initial_feed_dict = {}
        if not self.train_only:
            self.initial_feed_dict.update({self.is_training: True})
        self.initial_feed_dict.update(self.hyper_feed_dict)
        self.initial_feed_dict.update(self.train_dropout_dict)
        self.global_epochs += 1
        self.feed_dict = self.initial_feed_dict
        # Schedule the lr
        self.initial_feed_dict.update({self.global_epoch: self.global_epochs})


    def eval(self):
        """
        Enable the eval phase
        :return:
        """
        self.initial_feed_dict = {}
        if not self.train_only and not self.inference_only:
            self.initial_feed_dict.update({self.is_training: False})
        self.initial_feed_dict.update(self.hyper_feed_dict)
        self.initial_feed_dict.update(self.test_dropout_dict)
        self.feed_dict = self.initial_feed_dict


    def train_on_batch(self, data=None, run_train_only=False):
        """
        Train one batch on the data and do one time update.
        :param data: A tuple which will be sequentially fed into the input tensors.
        If the model is initialized with 'train_data_op' and 'val_data_op',
        'data' will be overrided and there is no need to specify 'data'.
        :return: loss, metrics in the self.loss and self.metrics collection.
        """
        if self.next_step_op is None:
            assert (len(data) <= len(self.input_tensors)), \
                "Too much data for inputs. Please assign data for each input layer"
            for i in range(len(data)):
                self.feed_dict.update({self.input_tensors[i]: data[i]})

        self.global_steps += self.ngpus
        self.feed_dict.update({self.global_step: self.global_steps})

        if run_train_only:
            self.sess.run(self.train_ops, feed_dict=self.feed_dict)
            return
        else:
            if self.global_steps % self.summary_display == 0:
                _, loss, metrics, summary_str = self.sess.run(
                    [self.train_ops, self.ngpus_loss_p, self.metric_tensors, self.summary_op], feed_dict=self.feed_dict)
                self.summary_writer.add_summary(summary_str, self.global_steps)
                return loss, metrics
            else:
                _, loss, metrics = self.sess.run(
                    [self.train_ops, self.ngpus_loss_p, self.metric_tensors], feed_dict=self.feed_dict)
                return loss, metrics

    def test_on_batch(self, data=None):
        """
        Test one batch data and return the loss and metrics
        :param data: A tuple which will be sequentially fed into the input tensors.
        If the model is initialized with 'train_data_op' and 'val_data_op',
        'data' will be overrided and there is no need to specify 'data'.
        :return: loss, metrics in the self.loss and self.metrics collection.
        """
        self.feed_dict = self.initial_feed_dict
        if self.is_training is None:
            assert (len(data) <= len(self.input_tensors)), \
                "Too much data for inputs. Please assign data for each input layer"
            for i in range(len(data)):
                self.feed_dict.update({self.input_tensors[i]: data[i]})
        self.feed_dict.update({self.global_step: self.global_steps})
        loss, metrics = self.sess.run([self.ngpus_loss_p, self.metric_tensors], feed_dict=self.feed_dict)
        return loss, metrics

    def predict(self, data):
        """
        Predict logits according to data.
        :param data: A tuple which will be sequentially fed into the input tensors.
        :return: predicted logits.
        """
        feed_dict = {}
        feed_dict.update(self.test_dropout_dict)
        if self.is_training is not None:
            self.feed_dict.update({self.is_training: False})
        else:
            assert (len(data) <= len(self.input_tensors)), \
                "Too much data for inputs. Please assign data for each input layer"
            for i in range(len(data)):
                self.feed_dict.update({self.input_tensors[i]: data[i]})
        if self.layerinfos is not None:
            feed_dict.update(self.layerinfos)
        return self.sess.run(self.output, feed_dict=feed_dict)

    def _load_model(self, path):
        """
        Load prototxt and build model
        :param path: prototxt path
        :return: None
        """
        input_slice_id = 0
        with open(path, 'r') as fp:
            while 1:
                ltype, paramdict = ParseBlock(fp)
                if ltype is None:
                    break
                if ltype == "Model":
                    name = ModelAssign(paramdict, 'name', 'demo')
                    self.pretrain_path = ModelAssign(paramdict, 'pretrain', None)
                    self.model_root = os.path.join("./models", name)
                    if not os.path.exists(self.model_root):
                        os.makedirs(self.model_root)
                    self.log_path = os.path.join("./models", name, 'log')
                    self.save_path = os.path.join("./models", name, 'model.ckpt')
                    self.drop_last_nvars = ModelAssign(paramdict, 'drop_last_nvars', 0)

                elif ltype in tflib.inputs_:
                    with tf.device("/cpu:0"):
                        layerobj = tflib.LayerObj[ltype](paramdict)
                        self.layers.append(layerobj)
                        self.layer_gpu_id.append("general")
                        name = ModelAssign(paramdict, 'name', 'demo')
                        #if self.next_step_op is not None:
                        #    tensor_dict = {"feed_tensor": self.next_step_op[input_slice_id]}
                        # input_slice_id += 1
                        #else:
                        #    tensor_dict = {"feed_tensor": None}
                        #layerobj(tensor_dict)
                        #self.input_tensors.append(layerobj.output_tensor)

                        self.input_collections.append(name)
                        #if self.batch_size is not None:
                        #    self.ngpu_input_tensors[name] = []
                        #    for i in range(self.ngpus):
                        #        self.ngpu_input_tensors[name].append(
                        #            layerobj.output_tensor[self.batch_size*i: self.batch_size*(i+1)])
                        #else:
                        #   if self.ngpus != 1:
                        #        if isinstance(layerobj.output_tensor, list):
                        #            self.ngpu_input_tensors = layerobj.output_tensor
                        #        else:
                        #            self.ngpu_input_tensors[name] = tf.split(layerobj.output_tensor, self.ngpus)
                        #    else:
                        #        self.ngpu_input_tensors[name] = [layerobj.output_tensor]
                        if self.ngpus == 1:
                            self.ngpu_input_tensors[name] = [self.next_step_op[input_slice_id]]
                        else:
                            self.ngpu_input_tensors[name] = []
                            for i in range(self.ngpus):
                                self.ngpu_input_tensors[name].append(self.next_step_op[i][input_slice_id])
                        input_slice_id += 1

                elif ltype in tflib.layers_:
                    name = ModelAssign(paramdict, 'name', 'demo')
                    for i in range(self.ngpus):
                        with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device=ps_device)):
                            tensor_dict = {}
                            input_name = RemoveListBrace(paramdict['input'])
                            input_name = input_name.split(',')
                            if len(input_name) == 1:
                                if input_name[0] in self.input_collections:
                                    tensor_dict['input_tensor'] = self.ngpu_input_tensors[input_name[0]][i]
                                else:
                                    tensor_dict['input_tensor'] = self.__getitem__(input_name[0], i)
                            else:
                                tensor_dict['input_tensor'] = []
                                for _names in input_name:
                                    if _names in self.input_collections:
                                        tensor_dict['input_tensor'].append(self.ngpu_input_tensors[_names][i])
                                    else:
                                        tensor_dict['input_tensor'].append(self.__getitem__(_names, i))
                            skip_from = ModelAssign(paramdict, 'skip_from', None)
                            if skip_from is not None:
                                skip_from = skip_from.strip('[')
                                skip_from = skip_from.strip(']')
                                skip_from = skip_from.split(',')
                                tensor_dict['skip_from_names'] = skip_from
                                skip_from_tensors = []
                                for skip_names in skip_from:
                                    skip_from_tensors.append(self.__getitem__(skip_names, i))
                                tensor_dict['skip_from'] = skip_from_tensors
                            tensor_dict['regularizer_strength'] = ModelAssign(paramdict, 'regularizer_strength', 1e-4)
                            tensor_dict['is_training'] = self.is_training
                            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                                layerobj = tflib.LayerObj[ltype](paramdict)
                                ret = layerobj(tensor_dict)

                        if ret is not None:
                            if 'layerinfo' in ret.keys():
                                self.layerinfos.update(ret['layerinfo'])
                            elif 'dropout' in ret.keys():
                                dropout = ModelAssign(paramdict, 'dropout', 0.0)
                                self.train_dropout_dict.update({ret['dropout']: dropout})
                                self.test_dropout_dict.update({ret['dropout']: 0.0})
                        self.layers.append(layerobj)
                        self.layer_gpu_id.append("gpu_%d" % i)
                        if i == 0:
                            self.num_trainable_parameters += layerobj.num_trainable_parameters
                            self.num_non_trainable_parameters += layerobj.num_non_trainable_parameters
                            self.flops += layerobj.MACs
                            if self.shared_trainable_parameters == 0:
                                self.shared_trainable_parameters += layerobj.shared_trainable_parameters
                            try:
                                self.mem_cost += layerobj.mem_cost
                                self.peak_mem = max(self.peak_mem, layerobj.peak_activation_mem)
                            except Exception:
                                pass

                elif ltype in tflib.losses_:
                    add_metric_tensors = None
                    for i in range(self.ngpus):
                        with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device=ps_device)):
                            tensor_dict = {}
                            input_name = paramdict['input']
                            tensor_dict['input'] = self.__getitem__(input_name, i)
                            label_name = paramdict['labels']
                            if label_name in self.input_collections:
                                tensor_dict['label'] = self.ngpu_input_tensors[label_name][i]
                            else:
                                tensor_dict['label'] = self.__getitem__(label_name, i)
                            if 'num_objects' in paramdict.keys():
                                num_objects_name = paramdict['num_objects']
                                tensor_dict['num_objects'] = self.__getitem__(num_objects_name, i)
                            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
                                layerobj = tflib.LayerObj[ltype](paramdict)
                                ret = layerobj(tensor_dict)

                                if ret is not None:
                                    if add_metric_tensors is None:
                                        add_metric_tensors = ret
                                    else:
                                        for key in ret.keys():
                                            add_metric_tensors[key] += ret[key] / self.ngpus
                                self.layers.append(layerobj)
                                self.layer_gpu_id.append("gpu_%d" % i)
                                self.loss = layerobj.output_tensor
                                if i == 0:
                                    self.ngpus_reg_losses = tf.losses.get_regularization_loss()
                                total_loss = self.loss
                                self.ngpus_loss.append(total_loss)
                                # Added for yolo

                    if add_metric_tensors is not None:
                        if 'cls_acc' in add_metric_tensors.keys():
                            self.metrics.append("cls_acc")
                            self.metric_tensors.append(add_metric_tensors['cls_acc'])
                            tf.summary.scalar("metrics/training_cls_acc", add_metric_tensors['cls_acc'])

                elif ltype in tflib.metrics_:
                    _metrics = []
                    # Only calculate the metric for the first tower.
                    for i in range(self.ngpus):
                        with tf.device(assign_to_device('/cpu:{}'.format(0), ps_device=ps_device)):
                            tensor_dict = {}
                            logits_name = paramdict['logits']
                            tensor_dict['logits'] = self.__getitem__(logits_name, i)
                            label_name = paramdict['labels']
                            if label_name in self.input_collections:
                                tensor_dict['labels'] = self.ngpu_input_tensors[label_name][i]
                            else:
                                tensor_dict['labels'] = self.__getitem__(label_name, i)
                            layerobj = tflib.LayerObj[ltype](paramdict)
                            layerobj(tensor_dict)
                            self.layers.append(layerobj)
                            self.layer_gpu_id.append("gpu_%d" % i)
                            _metrics.append(layerobj.output_tensor)

                    _metric = tf.reduce_mean(_metrics)
                    self.metric_tensors.append(_metric)
                    name = ModelAssign(paramdict, 'name', ltype)
                    tf.summary.scalar("metrics/training_%s" %name, _metric)

                elif ltype in tflib.outputs_:
                    with tf.device(assign_to_device('/gpu:{}'.format(0), ps_device=ps_device)):
                        tensor_dict = {}
                        tensor_dict['input'] = self.__getitem__(paramdict['input'], 0)
                        layerobj = tflib.LayerObj[ltype](paramdict)
                        layerobj(tensor_dict)
                        self.layers.append(layerobj)
                        self.layer_gpu_id.append("general")
                        self.output.append(layerobj.output_tensor)
            fp.close()


    def __getitem__(self, item, gpuid=0):
        """
        Fetch item with corresponding gpu_id.
        :param item: item name.
        :param gpuid: item gpu id.
        :return: A tensor.
        """
        l = len(self.layers)
        for i in range(l):
            if self.layers[i].name == item and \
                    (self.layer_gpu_id[i] == 'gpu_%d' %gpuid or self.layer_gpu_id[i] == 'general'):
                return self.layers[i].output_tensor

    def _load_solver(self, solver_txt):
        """
        Load solver and parsing the hyperparameters.
        Hyperparameters will be added to self.hyper_param_dict collection.
        :param solver_txt: solver_txt path
        :return: None
        """
        self.hyper_param_dict = {}
        with open(solver_txt, "r") as fp:
            line = fp.readline()
            while line != "":
                if line == "\n":
                    line = fp.readline(line)
                    continue
                key, value = SplitExpr(line[:-1])
                self.hyper_param_dict.update({key:value})
                line = fp.readline()
            fp.close()


    def load_weights(self, file_path):
        """
        Reload weights
        :param file_path: weight file path
        :return: None
        """
        try:
            self.saver.restore(self.sess, file_path)
        except Exception:
            print("Invalid weight file! Loading failed.")

    def save_weights(self, weight_path):
        """
        Save the model weights in 。ckpt format。
        :param weight_path: weight save path
        :return: None
        """
        self.saver.save(self.sess, weight_path)

    def save_to_graphpb(self, pb_log_dir, graph_name='graph.pb'):
        """
        Call this to save graphpb and weights
        :param pb_log_dir: graph pb logging dir
        :param graph_name: graph pb name
        :return:
        """
        tf.logging.warning("This function is deprecated. Use save_to_graphpb_and_freeze instead.")
        tf.train.write_graph(self.sess.graph_def, pb_log_dir, graph_name, False)


    def load_from_graphb(self, pb_graph_name='graph.pb'):
        """
        For reference only。 Do not call this method！
        :param pb_graph_name: graph pb name
        :return: None
        """
        tf.reset_default_graph()
        with gfile.FastGFile(pb_graph_name, 'rb') as fp:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')


    def close(self):
        """
        Close & clean up
        :return: None
        """
        tf.reset_default_graph()
        if self.summary_writer is not None:
            self.summary_writer.close()
        self.sess.close()
