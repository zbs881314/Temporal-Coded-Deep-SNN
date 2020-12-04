import numpy as np
import tensorflow as tf



class SNNLayer(object):
    def __init__(self, in_size, out_size, n_layer, w=None):

        layer_name = 'layer%s' % n_layer

        self.MAX_SPIKE_TIME = 1e5
        self.out_size = out_size
        self.in_size = in_size + 1
        if w is None:
            self.weight = tf.Variable(tf.concat((tf.random_uniform([self.in_size - 1, self.out_size], 0. / self.in_size,
                                                                   8. / self.in_size, tf.float32),
                                                 tf.zeros([1, self.out_size])), axis=0))
        else:
            self.weight = tf.Variable(w,dtype=tf.float32)

        tf.summary.histogram(layer_name + '/weights', self.weight)


    def forward(self,layer_in):
        batch_num = tf.shape(layer_in)[0]
        bias_layer_in = tf.ones([batch_num, 1])
        layer_in = tf.concat([layer_in, bias_layer_in], 1)
        _, input_sorted_indices = tf.nn.top_k(-layer_in, self.in_size, False)
        input_sorted = tf.batch_gather(layer_in, input_sorted_indices)
        input_sorted_outsize = tf.tile(tf.reshape(input_sorted, [batch_num, self.in_size, 1]), [1, 1, self.out_size])
        weight_sorted = tf.batch_gather(
            tf.tile(tf.reshape(self.weight, [1, self.in_size, self.out_size]), [batch_num, 1, 1]), input_sorted_indices)
        weight_input_mul = tf.multiply(weight_sorted, input_sorted_outsize)
        weight_sumed = tf.cumsum(weight_sorted, axis=1)
        weight_input_sumed = tf.cumsum(weight_input_mul, axis=1)
        out_spike_all = tf.divide(weight_input_sumed, tf.clip_by_value(weight_sumed - 1, 1e-10, 1e10))
        out_spike_ws = tf.where(weight_sumed < 1, self.MAX_SPIKE_TIME * tf.ones_like(out_spike_all), out_spike_all)
        out_spike_large = tf.where(out_spike_ws < input_sorted_outsize,
                                   self.MAX_SPIKE_TIME * tf.ones_like(out_spike_ws), out_spike_ws)
        input_sorted_outsize_slice = tf.slice(input_sorted_outsize, [0, 1, 0],
                                              [batch_num, self.in_size - 1, self.out_size])
        input_sorted_outsize_left = tf.concat(
            [input_sorted_outsize_slice, self.MAX_SPIKE_TIME * tf.ones([batch_num, 1, self.out_size])], 1)
        out_spike_valid = tf.where(out_spike_large > input_sorted_outsize_left,
                                   self.MAX_SPIKE_TIME * tf.ones_like(out_spike_large), out_spike_large)
        out_spike = tf.reduce_min(out_spike_valid, axis=1)
        return out_spike

    def w_sum_cost(self):
        threshold = 1.
        part1 = tf.subtract(threshold, tf.reduce_sum(self.weight, 0))
        part2 = tf.where(part1 > 0, part1, tf.zeros_like(part1))
        return tf.reduce_mean(part2)

    def l2_cost(self):
        w_sqr = tf.square(self.weight)
        return tf.reduce_mean(w_sqr)


class SCNNLayer(object):
    def __init__(self, kernel_size=3, in_channel=1, out_channel=1, strides=1, n_layer_new=1, w=None):
        self.MAX_SPIKE_TIME = 1e5
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.strides = strides
        self.kernel = SNNLayer(in_size=self.kernel_size * self.kernel_size * self.in_channel,
                                        out_size=self.out_channel, n_layer=n_layer_new, w=w)

    def forward(self, layer_in):
        input_size = tf.shape(layer_in)
        patches = tf.extract_image_patches(images=layer_in, ksizes=[1, self.kernel_size, self.kernel_size, 1],
                                           strides=[1, self.strides, self.strides, 1], rates=[1, 1, 1, 1],
                                           padding="SAME")
        patches_flatten = tf.reshape(patches,
                                     [input_size[0], -1, self.in_channel * self.kernel_size * self.kernel_size])
        patches_infpad = tf.where(tf.less(patches_flatten, 0.1),
                                  self.MAX_SPIKE_TIME * tf.ones_like(patches_flatten), patches_flatten)
        img_raw = tf.map_fn(self.kernel.forward, patches_infpad)
        img_reshaped = tf.reshape(img_raw,
                                  [input_size[0], tf.cast(tf.math.ceil(input_size[1] / self.strides), tf.int32),
                                   tf.cast(tf.math.ceil(input_size[2] / self.strides), tf.int32),
                                   self.out_channel])
        return img_reshaped


def loss_func(both):
    """
    function to calculate loss, refer to paper p.7, formula 11
    :param both: a tensor, it put both layer output and expected output together, its' shape
            is [batch_size,out_size*2], where the left part is layer output(real output), right part is
            the label of input(expected output), the tensor both should be looked like this:
            [[2.13,3.56,7.33,3.97,...0,0,1,0,...]
             [3.14,5.56,2.54,15.6,...0,0,0,1,...]...]
                ↑                   ↑
             layer output           label of input
    :return: a tensor, it is a scalar of loss
    """
    output = tf.slice(both, [0], [tf.cast(tf.shape(both)[0] / 2, tf.int32)])
    index = tf.slice(both, [tf.cast(tf.shape(both)[0] / 2, tf.int32)],
                     [tf.cast(tf.shape(both)[0] / 2, tf.int32)])
    z1 = tf.exp(tf.subtract(0., tf.reduce_sum(tf.multiply(output, index))))
    z2 = tf.reduce_sum(tf.exp(tf.subtract(0., output)))
    loss = tf.subtract(
        0., tf.log(
            tf.clip_by_value(tf.divide(
                z1, tf.clip_by_value(
                    z2, 1e-10, 1e10)), 1e-10, 1)))
    return loss

