import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np
import os
import SNN

input = tf.placeholder(tf.float32)
input_exp = tf.exp(input)
groundtruth = tf.placeholder(tf.float32)

layer_in = SNN.SNNLayer(784,800)
layer_out = SNN.SNNLayer(800,10)

layerin_out = layer_in.forward(input_exp)
layerout_out = layer_out.forward(layerin_out)


layerout_groundtruth = tf.concat([layerout_out,groundtruth],1)
loss = tf.reduce_mean(tf.map_fn(SNN.loss_func,layerout_groundtruth))

wsc = layer_in.w_sum_cost() + layer_out.w_sum_cost()
l2c = layer_in.l2_cost() + layer_out.l2_cost()

K = 100
K2 = 1e-3
learning_rate = 1e-4
TRAINING_BATCH = 10

SAVE_PATH = os.getcwd() + '/weight_mnist'

cost = loss + K*wsc + K2*l2c

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(cost)

config = tf.ConfigProto(
    device_count={'GPU': 1}
)
config.gpu_options.allow_growth = True
sess = tf.Session()
sess.run(tf.global_variables_initializer())

scale = 2
mnist = SNN.MnistData(path=["MNIST/train-images-idx3-ubyte","MNIST/train-labels-idx1-ubyte"])

print('training started')
step = 1
while(True):
    xs, ys = mnist.next_batch(TRAINING_BATCH, shuffle=True)
    xs = scale*(-xs + 1)
    [out,c,_] = sess.run([layerout_out,cost,train_op],{input:xs,groundtruth:ys})
    if step % 20 == 1:
        print('step '+repr(step) +', cost='+repr(c))
        w1 = sess.run(layer_in.weight)
        w2 = sess.run(layer_out.weight)
        np.save(SAVE_PATH + '1', w1)
        np.save(SAVE_PATH + '2', w2)
    step = step + 1

