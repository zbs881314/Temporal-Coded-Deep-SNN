import sys
import numpy as np
sys.path.append("..")
import SCNN1
import tensorflow as tf
import Cifar10_with_data_augmentation
import os



K = 100
K2 = 1e-2
TRAINING_BATCH = 128
TRAINING_EPOCHES = 500
learning_start = 1e-1
learning_end = 1e-5
lr_decay = (learning_end / learning_start) ** (1. / 500)



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


wsc1 = layer1.kernel.w_sum_cost()
wsc2 = layer2.kernel.w_sum_cost()
wsc3 = layer3.kernel.w_sum_cost()
wsc4 = layer4.kernel.w_sum_cost()
wsc5 = layer5.kernel.w_sum_cost()
wsc6 = layer6.kernel.w_sum_cost()
wsc7 = layer7.kernel.w_sum_cost()
wsc8 = layer8.kernel.w_sum_cost()
wsc9 = layer9.kernel.w_sum_cost()
wsc10 = layer10.kernel.w_sum_cost()
wsc11 = layer11.kernel.w_sum_cost()
wsc12 = layer12.kernel.w_sum_cost()
wsc13 = layer13.kernel.w_sum_cost()
wsc14 = layer14.w_sum_cost()
wsc15 = layer15.w_sum_cost()
wsc16 = layer16.w_sum_cost()
wsc17 = layer17.w_sum_cost()

wsc = wsc1 + wsc2 + wsc3 + wsc4 + wsc5 + wsc6 + wsc7 + wsc8 + wsc9 + wsc10 + wsc11 + wsc12 + wsc13 + wsc14 + wsc15 + wsc16 + wsc17

l21 = layer1.kernel.l2_cost()
l22 = layer2.kernel.l2_cost()
l23 = layer3.kernel.l2_cost()
l24 = layer4.kernel.l2_cost()
l25 = layer5.kernel.l2_cost()
l26 = layer6.kernel.l2_cost()
l27 = layer7.kernel.l2_cost()
l28 = layer8.kernel.l2_cost()
l29 = layer9.kernel.l2_cost()
l30 = layer10.kernel.l2_cost()
l31 = layer11.kernel.l2_cost()
l32 = layer12.kernel.l2_cost()
l33 = layer13.kernel.l2_cost()
l34 = layer14.l2_cost()
l35 = layer15.l2_cost()
l36 = layer16.l2_cost()
l37 = layer17.l2_cost()

l2 = l21+l22+l23+l24+l25+l26+l27+l28+l29+l30+l31+l32+l33+l34+l35+l36+l37


layerout_groundtruth = tf.concat([layerout17, output_real], 1)
output_loss = tf.reduce_mean(tf.map_fn(SCNN1.loss_func, layerout_groundtruth))

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
writer = tf.compat.v1.summary.FileWriter('./logs_cfar_a', sess.graph)

scale = 3

BATCH_SIZE = 128
SAVE_PATH = os.getcwd() + '/weight_CFAR_A'
cfar10 = Cifar10_with_data_augmentation.CFAR10()


for epoch in range(TRAINING_EPOCHES):
    print('epoch' + repr(epoch))

    for iteration in range(5000):
        xs, ys = cfar10.next_batch(batch_size=BATCH_SIZE, shuffle=True)
        xs = scale * xs
        [result,c,l,lo,_] = sess.run([merged,cost,output_loss,layerout17,train_op], {input_real: xs, output_real: ys, lr:learning_start * lr_decay ** epoch})
        step = sess.run(step_inc_op)
        writer.add_summary(result, step)
        if step % 10 == 0:
            print('step: ' + repr(step)+', cost= ' + repr(c)+', loss= ' + repr(l) + '\n' + 'layerout= '+repr(lo[0,:])+'\n'+'ys= '+repr(ys[0,:]))
            w1 = sess.run(layer1.kernel.weight)
            w2 = sess.run(layer2.kernel.weight)
            w3 = sess.run(layer3.kernel.weight)
            w4 = sess.run(layer4.kernel.weight)
            w5 = sess.run(layer5.kernel.weight)
            w6 = sess.run(layer6.kernel.weight)
            w7 = sess.run(layer7.kernel.weight)
            w8 = sess.run(layer8.kernel.weight)
            w9 = sess.run(layer9.kernel.weight)
            w10 = sess.run(layer10.kernel.weight)
            w11 = sess.run(layer11.kernel.weight)
            w12 = sess.run(layer12.kernel.weight)
            w13 = sess.run(layer13.kernel.weight)
            w14 = sess.run(layer14.weight)
            w15 = sess.run(layer15.weight)
            w16 = sess.run(layer16.weight)
            w17 = sess.run(layer17.weight)
            np.save(SAVE_PATH + '1', w1)
            np.save(SAVE_PATH + '2', w2)
            np.save(SAVE_PATH + '3', w3)
            np.save(SAVE_PATH + '4', w4)
            np.save(SAVE_PATH + '5', w5)
            np.save(SAVE_PATH + '6', w6)
            np.save(SAVE_PATH + '7', w7)
            np.save(SAVE_PATH + '8', w8)
            np.save(SAVE_PATH + '9', w9)
            np.save(SAVE_PATH + '10', w10)
            np.save(SAVE_PATH + '11', w11)
            np.save(SAVE_PATH + '12', w12)
            np.save(SAVE_PATH + '13', w13)
            np.save(SAVE_PATH + '14', w14)
            np.save(SAVE_PATH + '15', w15)
            np.save(SAVE_PATH + '16', w16)
            np.save(SAVE_PATH + '17', w17)


