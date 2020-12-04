import sys
import numpy as np
sys.path.append("..")
import SCNN
import tensorflow as tf
import Cifar10
import os


K = 100
K2 = 1e-2
TRAINING_BATCH = 128
TRAINING_EPOCHES = 100
learning_start = 1e-3
learning_end = 1e-5
lr_decay = (learning_end / learning_start) ** (1. / 100)



lr = tf.placeholder(tf.float32)
input_real = tf.placeholder(tf.float32)
output_real = tf.placeholder(tf.float32)

global_step = tf.Variable(1, dtype=tf.int64)
step_inc_op = tf.assign(global_step, global_step + 1)
'''
Here is a reshape, because TensorFlow DO NOT SUPPORT tf.extract_image_patches gradients operation for VARIABLE SIZE inputs
'''
input_real_resize = tf.reshape(tf.exp(input_real), [TRAINING_BATCH, 32, 32, 3])

try:
    w1 = np.load('weight_CFAR1.npy')
    w2 = np.load('weight_CFAR2.npy')
    w3 = np.load('weight_CFAR3.npy')
    w4 = np.load('weight_CFAR4.npy')
    w5 = np.load('weight_CFAR5.npy')
    print('Done')
    layer1 = SCNN.SCNNLayer(kernel_size=5, in_channel=3, out_channel=64, strides=2, n_layer_new=1, w=w1)
    layer2 = SCNN.SCNNLayer(kernel_size=5, in_channel=64, out_channel=32, strides=2, n_layer_new=2, w=w2)
    layer3 = SCNN.SCNNLayer(kernel_size=5, in_channel=32, out_channel=16, strides=2, n_layer_new=3, w=w3)
    layer4 = SCNN.SNNLayer(in_size=256, out_size=64, n_layer=4, w=w4)
    layer5 = SCNN.SNNLayer(in_size=64, out_size=10, n_layer=5, w=w5)
    print('Weight loaded!')
except:
    layer1 = SCNN.SCNNLayer(kernel_size=5, in_channel=3, out_channel=64, strides=2, n_layer_new=1)
    layer2 = SCNN.SCNNLayer(kernel_size=5, in_channel=64, out_channel=32, strides=2, n_layer_new=2)
    layer3 = SCNN.SCNNLayer(kernel_size=5, in_channel=32, out_channel=16, strides=2, n_layer_new=3)
    layer4 = SCNN.SNNLayer(in_size=256, out_size=64, n_layer=4)
    layer5 = SCNN.SNNLayer(in_size=64, out_size=10, n_layer=5)
    print('No weight file found, use random weight')

layerout1 = layer1.forward(input_real_resize)
print(layerout1.shape)
layerout2 = layer2.forward(layerout1)
print(layerout2.shape)
layerout3 = layer3.forward(layerout2)
print(layerout3.shape)
layerout4 = layer4.forward(tf.reshape(layerout3,[TRAINING_BATCH,256]))
layerout5 = layer5.forward(layerout4)


wsc1 = layer1.kernel.w_sum_cost()
wsc2 = layer2.kernel.w_sum_cost()
wsc3 = layer3.kernel.w_sum_cost()
wsc4 = layer4.w_sum_cost()
wsc5 = layer5.w_sum_cost()

wsc = wsc1 + wsc2 + wsc3 + wsc4 + wsc5

l21 = layer1.kernel.l2_cost()
l22 = layer2.kernel.l2_cost()
l23 = layer3.kernel.l2_cost()
l24 = layer4.l2_cost()
l25 = layer5.l2_cost()

l2 = l21+l22+l23+l24+l25


layerout_groundtruth = tf.concat([layerout5, output_real], 1)
output_loss = tf.reduce_mean(tf.map_fn(SCNN.loss_func, layerout_groundtruth))

cost = K*wsc + K2*l2 + output_loss

tf.summary.scalar('cost', cost)

opt = tf.train.AdamOptimizer(learning_rate=lr)
train_op = opt.minimize(cost)



config = tf.ConfigProto(
    device_count={'GPU': 1}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

tf.Graph.finalize(sess)

print("training started")

merged = tf.compat.v1.summary.merge_all()
writer = tf.compat.v1.summary.FileWriter('./logs_cfar_f2', sess.graph)

scale = 3

BATCH_SIZE = 128
SAVE_PATH = os.getcwd() + '/weight_CFAR'

cfar10 = Cifar10.CFAR10()


for epoch in range(TRAINING_EPOCHES):
    print('epoch' + repr(epoch))

    for iteration in range(5000):
        xs, ys = cfar10.next_batch(batch_size=BATCH_SIZE, shuffle=True)
        xs = scale * xs
        [result,c,l,lo,_] = sess.run([merged,cost,output_loss,layerout5,train_op], {input_real: xs, output_real: ys, lr:learning_start})
        step = sess.run(step_inc_op)
        writer.add_summary(result, step)
        if step % 10 == 0:
            print('step: ' + repr(step)+', cost= ' + repr(c)+', loss= ' + repr(l) + '\n' + 'layerout= '+repr(lo[0,:])+'\n'+'ys= '+repr(ys[0,:]))
            w1 = sess.run(layer1.kernel.weight)
            w2 = sess.run(layer2.kernel.weight)
            w3 = sess.run(layer3.kernel.weight)
            w4 = sess.run(layer4.weight)
            w5 = sess.run(layer5.weight)
            np.save(SAVE_PATH + '1', w1)
            np.save(SAVE_PATH + '2', w2)
            np.save(SAVE_PATH + '3', w3)
            np.save(SAVE_PATH + '4', w4)
            np.save(SAVE_PATH + '5', w5)


