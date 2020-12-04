import sys
import numpy as np
sys.path.append("..")
import SCNN1
import tensorflow as tf
import IMAGENET
from keras.utils import to_categorical


TESTING_BATCH = 10


input_real = tf.placeholder(tf.float32)
output_real = tf.placeholder(tf.float32)

global_step = tf.Variable(1, dtype=tf.int64)
step_inc_op = tf.assign(global_step, global_step + 1)
'''
Here is a reshape, because TensorFlow DO NOT SUPPORT tf.extract_image_patches gradients operation for VARIABLE SIZE inputs
'''
input_real_resize = tf.reshape(tf.exp(input_real), [TESTING_BATCH, 224, 224, 3])


w1 = np.load('weight_IMGN_A1.npy')
w2 = np.load('weight_IMGN_A2.npy')
w3 = np.load('weight_IMGN_A3.npy')
w4 = np.load('weight_IMGN_A4.npy')
w5 = np.load('weight_IMGN_A5.npy')
w6 = np.load('weight_IMGN_A6.npy')
w7 = np.load('weight_IMGN_A7.npy')
w8 = np.load('weight_IMGN_A8.npy')
w9 = np.load('weight_IMGN_A9.npy')
w10 = np.load('weight_IMGN_A10.npy')
w11 = np.load('weight_IMGN_A11.npy')
w12 = np.load('weight_IMGN_A12.npy')
w13 = np.load('weight_IMGN_A13.npy')
w14 = np.load('weight_IMGN_A14.npy')
w15 = np.load('weight_IMGN_A15.npy')
w16 = np.load('weight_IMGN_A16.npy')
w17 = np.load('weight_IMGN_A17.npy')
w18 = np.load('weight_IMGN_A18.npy')
w19 = np.load('weight_IMGN_A19.npy')
w20 = np.load('weight_IMGN_A20.npy')
w21 = np.load('weight_IMGN_A21.npy')
w22 = np.load('weight_IMGN_A22.npy')
w23 = np.load('weight_IMGN_A23.npy')
w24 = np.load('weight_IMGN_A24.npy')
w25 = np.load('weight_IMGN_A25.npy')
w26 = np.load('weight_IMGN_A26.npy')
w27 = np.load('weight_IMGN_A27.npy')
w28 = np.load('weight_IMGN_A28.npy')
w29 = np.load('weight_IMGN_A29.npy')
w30 = np.load('weight_IMGN_A30.npy')
w31 = np.load('weight_IMGN_A31.npy')
w32 = np.load('weight_IMGN_A32.npy')
w33 = np.load('weight_IMGN_A33.npy')
w34 = np.load('weight_IMGN_A34.npy')
w35 = np.load('weight_IMGN_A35.npy')
w36 = np.load('weight_IMGN_A36.npy')
w37 = np.load('weight_IMGN_A37.npy')
w38 = np.load('weight_IMGN_A38.npy')
w39 = np.load('weight_IMGN_A39.npy')
w40 = np.load('weight_IMGN_A40.npy')
w41 = np.load('weight_IMGN_A41.npy')
w42 = np.load('weight_IMGN_A42.npy')
w43 = np.load('weight_IMGN_A43.npy')
w44 = np.load('weight_IMGN_A44.npy')
w45 = np.load('weight_IMGN_A45.npy')
w46 = np.load('weight_IMGN_A46.npy')
w47 = np.load('weight_IMGN_A47.npy')
w48 = np.load('weight_IMGN_A48.npy')
w49 = np.load('weight_IMGN_A49.npy')
w50 = np.load('weight_IMGN_A50.npy')
w51 = np.load('weight_IMGN_A51.npy')
w52 = np.load('weight_IMGN_A52.npy')
w53 = np.load('weight_IMGN_A53.npy')
w54 = np.load('weight_IMGN_A54.npy')
w55 = np.load('weight_IMGN_A55.npy')
w56 = np.load('weight_IMGN_A56.npy')
w57 = np.load('weight_IMGN_A57.npy')
w58 = np.load('weight_IMGN_A58.npy')
print('Done')
layer1 = SCNN1.SCNNLayer(kernel_size=7, in_channel=3, out_channel=64, strides=2, n_layer_new=1, w=w1)
layer2 = SCNN1.SCNNLayer(kernel_size=1, in_channel=64, out_channel=64, strides=1, n_layer_new=2, w=w2)
layer3 = SCNN1.SCNNLayer(kernel_size=3, in_channel=64, out_channel=192, strides=1, n_layer_new=3, w=w3)

layer4 = SCNN1.SCNNLayer(kernel_size=1, in_channel=192, out_channel=64, strides=1, n_layer_new=4, w=w4)
layer5 = SCNN1.SCNNLayer(kernel_size=1, in_channel=192, out_channel=96, strides=1, n_layer_new=5, w=w5)
layer6 = SCNN1.SCNNLayer(kernel_size=3, in_channel=96, out_channel=128, strides=1, n_layer_new=6, w=w6)
layer7 = SCNN1.SCNNLayer(kernel_size=1, in_channel=192, out_channel=16, strides=1, n_layer_new=7, w=w7)
layer8 = SCNN1.SCNNLayer(kernel_size=5, in_channel=16, out_channel=32, strides=1, n_layer_new=8, w=w8)
layer9 = SCNN1.SCNNLayer(kernel_size=1, in_channel=192, out_channel=32, strides=1, n_layer_new=9, w=w9)

layer10 = SCNN1.SCNNLayer(kernel_size=1, in_channel=256, out_channel=128, strides=1, n_layer_new=10, w=w10)
layer11 = SCNN1.SCNNLayer(kernel_size=1, in_channel=256, out_channel=128, strides=1, n_layer_new=11, w=w11)
layer12 = SCNN1.SCNNLayer(kernel_size=3, in_channel=128, out_channel=192, strides=1, n_layer_new=12, w=w12)
layer13 = SCNN1.SCNNLayer(kernel_size=1, in_channel=256, out_channel=32, strides=1, n_layer_new=13, w=w13)
layer14 = SCNN1.SCNNLayer(kernel_size=5, in_channel=32, out_channel=96, strides=1, n_layer_new=14, w=w14)
layer15 = SCNN1.SCNNLayer(kernel_size=1, in_channel=256, out_channel=64, strides=1, n_layer_new=15, w=w15)

layer16 = SCNN1.SCNNLayer(kernel_size=1, in_channel=480, out_channel=192, strides=1, n_layer_new=16, w=w16)
layer17 = SCNN1.SCNNLayer(kernel_size=1, in_channel=480, out_channel=96, strides=1, n_layer_new=17, w=w17)
layer18 = SCNN1.SCNNLayer(kernel_size=3, in_channel=96, out_channel=208, strides=1, n_layer_new=18, w=w18)
layer19 = SCNN1.SCNNLayer(kernel_size=1, in_channel=480, out_channel=16, strides=1, n_layer_new=19, w=w19)
layer20 = SCNN1.SCNNLayer(kernel_size=5, in_channel=16, out_channel=48, strides=1, n_layer_new=20, w=w20)
layer21 = SCNN1.SCNNLayer(kernel_size=1, in_channel=480, out_channel=64, strides=1, n_layer_new=21, w=w21)

layer22 = SCNN1.SCNNLayer(kernel_size=1, in_channel=512, out_channel=160, strides=1, n_layer_new=22, w=w22)
layer23 = SCNN1.SCNNLayer(kernel_size=1, in_channel=512, out_channel=112, strides=1, n_layer_new=23, w=w23)
layer24 = SCNN1.SCNNLayer(kernel_size=3, in_channel=112, out_channel=224, strides=1, n_layer_new=24, w=w24)
layer25 = SCNN1.SCNNLayer(kernel_size=1, in_channel=512, out_channel=24, strides=1, n_layer_new=25, w=w25)
layer26 = SCNN1.SCNNLayer(kernel_size=5, in_channel=24, out_channel=64, strides=1, n_layer_new=26, w=w26)
layer27 = SCNN1.SCNNLayer(kernel_size=1, in_channel=512, out_channel=64, strides=1, n_layer_new=27, w=w27)

layer28 = SCNN1.SCNNLayer(kernel_size=1, in_channel=512, out_channel=128, strides=1, n_layer_new=28, w=w28)
layer29 = SCNN1.SCNNLayer(kernel_size=1, in_channel=512, out_channel=128, strides=1, n_layer_new=29, w=w29)
layer30 = SCNN1.SCNNLayer(kernel_size=3, in_channel=128, out_channel=256, strides=1, n_layer_new=30, w=w30)
layer31 = SCNN1.SCNNLayer(kernel_size=1, in_channel=512, out_channel=24, strides=1, n_layer_new=31, w=w31)
layer32 = SCNN1.SCNNLayer(kernel_size=5, in_channel=24, out_channel=64, strides=1, n_layer_new=32, w=w32)
layer33 = SCNN1.SCNNLayer(kernel_size=1, in_channel=512, out_channel=64, strides=1, n_layer_new=33, w=w33)

layer34 = SCNN1.SCNNLayer(kernel_size=1, in_channel=512, out_channel=112, strides=1, n_layer_new=34, w=w34)
layer35 = SCNN1.SCNNLayer(kernel_size=1, in_channel=512, out_channel=144, strides=1, n_layer_new=35, w=w35)
layer36 = SCNN1.SCNNLayer(kernel_size=3, in_channel=144, out_channel=288, strides=1, n_layer_new=36, w=w36)
layer37 = SCNN1.SCNNLayer(kernel_size=1, in_channel=512, out_channel=32, strides=1, n_layer_new=37, w=w37)
layer38 = SCNN1.SCNNLayer(kernel_size=5, in_channel=32, out_channel=64, strides=1, n_layer_new=38, w=w38)
layer39 = SCNN1.SCNNLayer(kernel_size=1, in_channel=512, out_channel=64, strides=1, n_layer_new=39, w=w39)

layer40 = SCNN1.SCNNLayer(kernel_size=1, in_channel=528, out_channel=256, strides=1, n_layer_new=40, w=w40)
layer41 = SCNN1.SCNNLayer(kernel_size=1, in_channel=528, out_channel=160, strides=1, n_layer_new=41, w=w41)
layer42 = SCNN1.SCNNLayer(kernel_size=3, in_channel=160, out_channel=320, strides=1, n_layer_new=42, w=w42)
layer43 = SCNN1.SCNNLayer(kernel_size=1, in_channel=528, out_channel=32, strides=1, n_layer_new=43, w=w43)
layer44 = SCNN1.SCNNLayer(kernel_size=5, in_channel=32, out_channel=128, strides=1, n_layer_new=44, w=w44)
layer45 = SCNN1.SCNNLayer(kernel_size=1, in_channel=528, out_channel=128, strides=1, n_layer_new=45, w=w45)

layer46 = SCNN1.SCNNLayer(kernel_size=1, in_channel=832, out_channel=256, strides=1, n_layer_new=46, w=w46)
layer47 = SCNN1.SCNNLayer(kernel_size=1, in_channel=832, out_channel=160, strides=1, n_layer_new=47, w=w47)
layer48 = SCNN1.SCNNLayer(kernel_size=3, in_channel=160, out_channel=320, strides=1, n_layer_new=48, w=w48)
layer49 = SCNN1.SCNNLayer(kernel_size=1, in_channel=832, out_channel=32, strides=1, n_layer_new=49, w=w49)
layer50 = SCNN1.SCNNLayer(kernel_size=5, in_channel=32, out_channel=128, strides=1, n_layer_new=50, w=w50)
layer51 = SCNN1.SCNNLayer(kernel_size=1, in_channel=832, out_channel=128, strides=1, n_layer_new=51, w=w51)

layer52 = SCNN1.SCNNLayer(kernel_size=1, in_channel=832, out_channel=384, strides=1, n_layer_new=52, w=w52)
layer53 = SCNN1.SCNNLayer(kernel_size=1, in_channel=832, out_channel=192, strides=1, n_layer_new=53, w=w53)
layer54 = SCNN1.SCNNLayer(kernel_size=3, in_channel=192, out_channel=384, strides=1, n_layer_new=54, w=w54)
layer55 = SCNN1.SCNNLayer(kernel_size=1, in_channel=832, out_channel=48, strides=1, n_layer_new=55, w=w55)
layer56 = SCNN1.SCNNLayer(kernel_size=5, in_channel=48, out_channel=128, strides=1, n_layer_new=56, w=w56)
layer57 = SCNN1.SCNNLayer(kernel_size=1, in_channel=832, out_channel=128, strides=1, n_layer_new=57, w=w57)
layer58 = SCNN1.SNNLayer(in_size=1024, out_size=1000, n_layer=58, w=w58)
print('Weight loaded!')


conv1_7x7_s2 = layer1.forward(input_real_resize)
pool1_3x3_s2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_7x7_s2)
pool1_norm1 = SCNN1.LRN(name='pool1/norm1')(pool1_3x3_s2)

conv2_3x3_reduce = layer2.forward(pool1_norm1)
conv2_3x3 = layer3.forward(conv2_3x3_reduce)
pool2_3x3_s2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2_3x3)

inception_3a_1x1 = layer4.forward(pool2_3x3_s2)
inception_3a_3x3_reduce = layer5.forward(pool2_3x3_s2)
inception_3a_3x3 = layer6.forward(inception_3a_3x3_reduce)
inception_3a_5x5_reduce = layer7.forward(pool2_3x3_s2)
inception_3a_5x5 = layer8.forward(inception_3a_5x5_reduce)
inception_3a_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(pool2_3x3_s2)
inception_3a_pool_proj = layer9.forward(inception_3a_pool)
inception_3a_output = tf.concat([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj], axis=3)

inception_3b_1x1 = layer10.forward(inception_3a_output)
inception_3b_3x3_reduce = layer11.forward(inception_3a_output)
inception_3b_3x3 = layer12.forward(inception_3b_3x3_reduce)
inception_3b_5x5_reduce = layer13.forward(inception_3a_output)
inception_3b_5x5 = layer14.forward(inception_3b_5x5_reduce)
inception_3b_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inception_3a_output)
inception_3b_pool_proj = layer15.forward(inception_3b_pool)
inception_3b_output = tf.concat([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj], axis=3)

inception_3b_output_zero_pad = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(inception_3b_output)
pool3_3x3_s2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(inception_3b_output_zero_pad)

inception_4a_1x1 = layer16.forward(pool3_3x3_s2)
inception_4a_3x3_reduce = layer17.forward(pool3_3x3_s2)
inception_4a_3x3 = layer18.forward(inception_4a_3x3_reduce)
inception_4a_5x5_reduce = layer19.forward(pool3_3x3_s2)
inception_4a_5x5 = layer20.forward(inception_4a_5x5_reduce)
inception_4a_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(pool3_3x3_s2)
inception_4a_pool_proj = layer21.forward(inception_4a_pool)
inception_4a_output = tf.concat([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj], axis=3)

inception_4b_1x1 = layer22.forward(inception_4a_output)
inception_4b_3x3_reduce = layer23.forward(inception_4a_output)
inception_4b_3x3 = layer24.forward(inception_4b_3x3_reduce)
inception_4b_5x5_reduce = layer25.forward(inception_4a_output)
inception_4b_5x5 = layer26.forward(inception_4b_5x5_reduce)
inception_4b_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inception_4a_output)
inception_4b_pool_proj = layer27.forward(inception_4b_pool)
inception_4b_output = tf.concat([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj], axis=3)

inception_4c_1x1 = layer28.forward(inception_4b_output)
inception_4c_3x3_reduce = layer29.forward(inception_4b_output)
inception_4c_3x3 = layer30.forward(inception_4c_3x3_reduce)
inception_4c_5x5_reduce = layer31.forward(inception_4b_output)
inception_4c_5x5 = layer32.forward(inception_4c_5x5_reduce)
inception_4c_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inception_4b_output)
inception_4c_pool_proj = layer33.forward(inception_4c_pool)
inception_4c_output = tf.concat([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj], axis=3)

inception_4d_1x1 = layer34.forward(inception_4c_output)
inception_4d_3x3_reduce = layer35.forward(inception_4c_output)
inception_4d_3x3 = layer36.forward(inception_4d_3x3_reduce)
inception_4d_5x5_reduce = layer37.forward(inception_4c_output)
inception_4d_5x5 = layer38.forward(inception_4d_5x5_reduce)
inception_4d_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inception_4c_output)
inception_4d_pool_proj = layer39.forward(inception_4d_pool)
inception_4d_output = tf.concat([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj], axis=3)

inception_4e_1x1 = layer40.forward(inception_4d_output)
inception_4e_3x3_reduce = layer41.forward(inception_4d_output)
inception_4e_3x3 = layer42.forward(inception_4e_3x3_reduce)
inception_4e_5x5_reduce = layer43.forward(inception_4d_output)
inception_4e_5x5 = layer44.forward(inception_4e_5x5_reduce)
inception_4e_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inception_4d_output)
inception_4e_pool_proj = layer45.forward(inception_4e_pool)
inception_4e_output = tf.concat([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj], axis=3)

inception_4e_output_zero_pad = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(inception_4e_output)
pool4_3x3_s2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(inception_4e_output_zero_pad)

inception_5a_1x1 = layer46.forward(pool4_3x3_s2)
inception_5a_3x3_reduce = layer47.forward(pool4_3x3_s2)
inception_5a_3x3 = layer48.forward(inception_5a_3x3_reduce)
inception_5a_5x5_reduce = layer49.forward(pool4_3x3_s2)
inception_5a_5x5 = layer50.forward(inception_5a_5x5_reduce)
inception_5a_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(pool4_3x3_s2)
inception_5a_pool_proj = layer51.forward(inception_5a_pool)
inception_5a_output = tf.concat([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj], axis=3)

inception_5b_1x1 = layer52.forward(inception_5a_output)
inception_5b_3x3_reduce = layer53.forward(inception_5a_output)
inception_5b_3x3 = layer54.forward(inception_5b_3x3_reduce)
inception_5b_5x5_reduce = layer55.forward(inception_5a_output)
inception_5b_5x5 = layer56.forward(inception_5b_5x5_reduce)
inception_5b_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inception_5a_output)
inception_5b_pool_proj = layer57.forward(inception_5b_pool)
inception_5b_output = tf.concat([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj], axis=3)

pool5_7x7_s1 = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(inception_5b_output)

final_output = layer58.forward(tf.reshape(pool5_7x7_s1,[TESTING_BATCH,1024]))



config = tf.ConfigProto(
    device_count={'GPU': 1}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


print("Testing started")

scale = 3

BATCH_SIZE = 10

ImageN = IMAGENET.ImageData(path=["datasets/train_y.npy", "datasets/label_y.npy"])


accuracy = []
while(True):
    xs, ys = ImageN.next_batch(batch_size=BATCH_SIZE, shuffle=False)
    xs = scale * xs
    lo = sess.run(final_output, {input_real: xs, output_real: ys})
    layerout = np.argmin(lo, axis=1)
    layerout = to_categorical(layerout, 1000)
    accurate = 0
    for i in range(len(ys)):
        if (layerout[i] == ys[i]).all():
            accurate = accurate + 1
    accurate = accurate / 10.
    # print(accurate)
    accuracy.append(accurate)
    step = sess.run(step_inc_op)
    if step % 10 == 0:
        accurate1 = tf.reduce_mean(accuracy)
        print('Step: ' + repr(step) + ', ' + 'Accurate: ' + repr(sess.run(accurate1)))

    if step == 10000:
        break
