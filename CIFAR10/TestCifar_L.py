import sys
import numpy as np
sys.path.append("..")
import SCNN1
import tensorflow as tf
import CFAR10
from keras.utils import to_categorical



TRAINING_BATCH = 128
input_real = tf.placeholder(tf.float32)
output_real = tf.placeholder(tf.float32)

global_step = tf.Variable(1, dtype=tf.int64)
step_inc_op = tf.assign(global_step, global_step + 1)
'''
Here is a reshape, because TensorFlow DO NOT SUPPORT tf.extract_image_patches gradients operation for VARIABLE SIZE inputs
'''
input_real_resize = tf.reshape(tf.exp(input_real), [TRAINING_BATCH, 32, 32, 3])

try:
    w1 = np.load('weight_CFAR_A1.npy')
    w2 = np.load('weight_CFAR_A2.npy')
    w3 = np.load('weight_CFAR_A3.npy')
    w4 = np.load('weight_CFAR_A4.npy')
    w5 = np.load('weight_CFAR_A5.npy')
    w6 = np.load('weight_CFAR_A6.npy')
    w7 = np.load('weight_CFAR_A7.npy')
    w8 = np.load('weight_CFAR_A8.npy')
    w9 = np.load('weight_CFAR_A9.npy')
    w10 = np.load('weight_CFAR_A10.npy')
    w11 = np.load('weight_CFAR_A11.npy')
    w12 = np.load('weight_CFAR_A12.npy')
    w13 = np.load('weight_CFAR_A13.npy')
    w14 = np.load('weight_CFAR_A14.npy')
    w15 = np.load('weight_CFAR_A15.npy')
    w16 = np.load('weight_CFAR_A16.npy')
    w17 = np.load('weight_CFAR_A17.npy')
    print('Done')
    layer1 = SCNN1.SCNNLayer(kernel_size=3, in_channel=3, out_channel=64, strides=1, n_layer_new=1, w=w1)
    layer2 = SCNN1.SCNNLayer(kernel_size=3, in_channel=64, out_channel=64, strides=1, n_layer_new=2, w=w2)
    layer3 = SCNN1.SCNNLayer(kernel_size=3, in_channel=64, out_channel=128, strides=1, n_layer_new=3, w=w3)
    layer4 = SCNN1.SCNNLayer(kernel_size=3, in_channel=128, out_channel=128, strides=1, n_layer_new=4, w=w4)
    layer5 = SCNN1.SCNNLayer(kernel_size=3, in_channel=128, out_channel=256, strides=1, n_layer_new=5, w=w5)
    layer6 = SCNN1.SCNNLayer(kernel_size=3, in_channel=256, out_channel=256, strides=1, n_layer_new=6, w=w6)
    layer7 = SCNN1.SCNNLayer(kernel_size=3, in_channel=256, out_channel=256, strides=1, n_layer_new=7, w=w7)
    layer8 = SCNN1.SCNNLayer(kernel_size=3, in_channel=256, out_channel=512, strides=1, n_layer_new=8, w=w8)
    layer9 = SCNN1.SCNNLayer(kernel_size=3, in_channel=512, out_channel=512, strides=1, n_layer_new=9, w=w9)
    layer10 = SCNN1.SCNNLayer(kernel_size=3, in_channel=512, out_channel=512, strides=1, n_layer_new=10, w=w10)
    layer11 = SCNN1.SCNNLayer(kernel_size=3, in_channel=512, out_channel=1024, strides=1, n_layer_new=11, w=w11)
    layer12 = SCNN1.SCNNLayer(kernel_size=3, in_channel=1024, out_channel=1024, strides=1, n_layer_new=12, w=w12)
    layer13 = SCNN1.SCNNLayer(kernel_size=3, in_channel=1024, out_channel=1024, strides=1, n_layer_new=13, w=w13)
    layer14 = SCNN1.SNNLayer(in_size=1024, out_size=4096, n_layer=14, w=w14)
    layer15 = SCNN1.SNNLayer(in_size=4096, out_size=4096, n_layer=15, w=w15)
    layer16 = SCNN1.SNNLayer(in_size=4096, out_size=512, n_layer=16, w=w16)
    layer17 = SCNN1.SNNLayer(in_size=512, out_size=10, n_layer=17, w=w17)
    print('Weight loaded!')
except:
    layer1 = SCNN1.SCNNLayer(kernel_size=3, in_channel=3, out_channel=64, strides=1, n_layer_new=1)
    layer2 = SCNN1.SCNNLayer(kernel_size=3, in_channel=64, out_channel=64, strides=1, n_layer_new=2)
    layer3 = SCNN1.SCNNLayer(kernel_size=3, in_channel=64, out_channel=128, strides=1, n_layer_new=3)
    layer4 = SCNN1.SCNNLayer(kernel_size=3, in_channel=128, out_channel=128, strides=1, n_layer_new=4)
    layer5 = SCNN1.SCNNLayer(kernel_size=3, in_channel=128, out_channel=256, strides=1, n_layer_new=5)
    layer6 = SCNN1.SCNNLayer(kernel_size=3, in_channel=256, out_channel=256, strides=1, n_layer_new=6)
    layer7 = SCNN1.SCNNLayer(kernel_size=3, in_channel=256, out_channel=256, strides=1, n_layer_new=7)
    layer8 = SCNN1.SCNNLayer(kernel_size=3, in_channel=256, out_channel=512, strides=1, n_layer_new=8)
    layer9 = SCNN1.SCNNLayer(kernel_size=3, in_channel=512, out_channel=512, strides=1, n_layer_new=9)
    layer10 = SCNN1.SCNNLayer(kernel_size=3, in_channel=512, out_channel=512, strides=1, n_layer_new=10)
    layer11 = SCNN1.SCNNLayer(kernel_size=3, in_channel=512, out_channel=1024, strides=1, n_layer_new=11)
    layer12 = SCNN1.SCNNLayer(kernel_size=3, in_channel=1024, out_channel=1024, strides=1, n_layer_new=12)
    layer13 = SCNN1.SCNNLayer(kernel_size=3, in_channel=1024, out_channel=1024, strides=1, n_layer_new=13)
    layer14 = SCNN1.SNNLayer(in_size=1024, out_size=4096, n_layer=14)
    layer15 = SCNN1.SNNLayer(in_size=4096, out_size=4096, n_layer=15)
    layer16 = SCNN1.SNNLayer(in_size=4096, out_size=512, n_layer=16)
    layer17 = SCNN1.SNNLayer(in_size=512, out_size=10, n_layer=17)
    print('No weight file found, use random weight')

layerout1 = layer1.forward(input_real_resize)
print(layerout1.shape)
layerout2 = layer2.forward(layerout1)
print(layerout2.shape)
maxpool1 = SCNN1.max_pool_layer(layerout2, (2, 2), (2, 2), 'maxpool1')

layerout3 = layer3.forward(maxpool1)
print(layerout3.shape)
layerout4 = layer4.forward(layerout3)
maxpool2 = SCNN1.max_pool_layer(layerout4, (2, 2), (2, 2), 'maxpool2')

layerout5 = layer5.forward(maxpool2)
layerout6 = layer6.forward(layerout5)
layerout7 = layer7.forward(layerout6)
maxpool3 = SCNN1.max_pool_layer(layerout7, (2, 2), (2, 2), 'maxpool3')

layerout8 = layer8.forward(maxpool3)
layerout9 = layer9.forward(layerout8)
layerout10 = layer10.forward(layerout9)
maxpool4 = SCNN1.max_pool_layer(layerout10, (2, 2), (2, 2), 'maxpool4')

layerout11 = layer11.forward(maxpool4)
layerout12 = layer12.forward(layerout11)
layerout13 = layer13.forward(layerout12)
maxpool5 = SCNN1.max_pool_layer(layerout13, (2, 2), (2, 2), 'maxpool5')

layerout14 = layer14.forward(tf.reshape(maxpool5,[TRAINING_BATCH,1024]))
layerout15 = layer15.forward(layerout14)
layerout16 = layer16.forward(layerout15)
layerout17 = layer17.forward(layerout16)


config = tf.ConfigProto(
    device_count={'GPU': 1}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


print("Testing started")

scale = 3

BATCH_SIZE = 128

cfar10 = CFAR10.CifarData(path=["CIFAR/train_y.npy", "CIFAR/label_y.npy"])


accuracy = []
while(True):
    xs, ys = cfar10.next_batch(batch_size=BATCH_SIZE, shuffle=False)
    xs = scale * xs
    lo = sess.run(layerout17, {input_real: xs, output_real: ys})
    layerout = np.argmin(lo, axis=1)
    layerout = to_categorical(layerout, 10)
    accurate = 0
    for i in range(len(ys)):
        if (layerout[i] == ys[i]).all():
            accurate = accurate + 1
    accurate = accurate / 128.
    # print(accurate)
    accuracy.append(accurate)
    step = sess.run(step_inc_op)
    if step % 10 == 0:
        accurate1 = tf.reduce_mean(accuracy)
        print('Step: ' + repr(step) + ', ' + 'Accurate: ' + repr(sess.run(accurate1)))

    if step == 78:
        break
