import tensorflow as tf

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        grads = []
        for g, _ in grad_and_vars:
            grads.append(g)

        grad = tf.reduce_mean(grads, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def average_and_quantize_gradients(tower_grads, min=-5, max=5, T=tf.float16):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        grads = []
        for g, _ in grad_and_vars:
            q_g = tf.quantization.quantize(g, min, max, T)
            grads.append(q_g)

        grad = tf.reduce_mean(grads, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

