import sys
import numpy as np
sys.path.append("..")
import SNN
import tensorflow as tf
import os
from keras.utils import to_categorical


TRAINING_BATCH = 10


input_real = tf.placeholder(tf.float32)
output_real = tf.placeholder(tf.float32)

global_step = tf.Variable(1, dtype=tf.int64)
step_inc_op = tf.assign(global_step, global_step + 1)
'''
Here is a reshape, because TensorFlow DO NOT SUPPORT tf.extract_image_patches gradients operation for VARIABLE SIZE inputs
'''
input_real_resize = tf.reshape(tf.exp(input_real),[TRAINING_BATCH,28,28,1])

try:
    w1 = np.load('weight_scnn1.npy')
    w2 = np.load('weight_scnn2.npy')
    w3 = np.load('weight_scnn3.npy')
    print('Done')
    layer1 = SNN.SCNNLayer(kernel_size=5,in_channel=1,out_channel=32,strides=2, w=w1)
    layer2 = SNN.SCNNLayer(kernel_size=5,in_channel=32,out_channel=16,strides=2, w=w2)
    layer3 = SNN.SNNLayer(in_size=784,out_size=10, w=w3)
    print('Weight loaded!')
except:
    layer1 = SNN.SCNNLayer(kernel_size=5, in_channel=1, out_channel=32, strides=2)
    layer2 = SNN.SCNNLayer(kernel_size=5, in_channel=32, out_channel=16, strides=2)
    layer3 = SNN.SNNLayer(in_size=784, out_size=10)
    print('No weight file found, use random weight')
layerout1 = layer1.forward(input_real_resize)
layerout2 = layer2.forward(layerout1)
layerout3 = layer3.forward(tf.reshape(layerout2,[TRAINING_BATCH,784]))


nnout = tf.log(layerout3)


config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
tf.Graph.finalize(sess)


print("testing started")

scale = 3
mnist = SNN.MnistData(path=["MNIST/t10k-images-idx3-ubyte","MNIST/t10k-labels-idx1-ubyte"])


accuracy = []
while(True):

    xs, ys = mnist.next_batch(TRAINING_BATCH, shuffle=False)
    xs_new = scale * (-xs + 1)
    xs = np.reshape(xs_new, [-1, 28, 28, 1])
    [lo, out] = sess.run([layerout3, nnout], {input_real: xs, output_real: ys})
    layerout = np.argmin(lo, axis=1)
    layerout = to_categorical(layerout, 10)
    accurate = 0
    for i in range(len(ys)):
        if (layerout[i] == ys[i]).all():
            accurate = accurate + 1
    accurate = accurate / 10
    # print(accurate)
    accuracy.append(accurate)
    step = sess.run(step_inc_op)
    if step % 10 == 0:
        accurate1 = tf.reduce_mean(accuracy)
        print('Step: ' + repr(step) + ', ' + 'Accurate: ' + repr(sess.run(accurate1)))

    if step == 1001:
        break
