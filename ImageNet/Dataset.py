import os
import numpy as np
import csv
import imageio
from keras.utils import to_categorical


def take_central(img_input):
    if img_input.shape[0] > img_input.shape[1]:
        x_begin = 17
        y_begin = np.int(np.floor(img_input.shape[0]/2) - 112)
    else:
        y_begin = 17
        x_begin = np.int(np.floor(img_input.shape[1]/2) - 112)
    img_output = img_input[y_begin:(y_begin+224), x_begin:(x_begin+224), :]
    return img_output


test_images = 100000
data_folder = 'data/imagenet/'

# Load images for test set
y_test_file = []
y_test = np.zeros((test_images, 224, 224, 3)).astype('uint8')
for id in range(test_images):
    test_file = 'ILSVRC2012_val_0000' + str(id+1).zfill(4) + '.JPEG'
    y_test_file.append(test_file)
    img_path = os.path.join(data_folder, test_file)
    img = imageio.imread(img_path)
    y_test[id, :, :, :] = take_central(img)
y_test_preprocess = (y_test.astype('float32') - 0) / 255

print(y_test_preprocess)


label_name = 'data/imagenet/labels_testing.txt'
testing_labels = dict()
with open(label_name, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        testing_labels[row[0]] = row[1]

# Get labels for the test set by index
labels_test = np.zeros(test_images)
for i in range(test_images):
    labels_test[i] = int(testing_labels[y_test_file[i]])
label_test = to_categorical(labels_test, 1000)


img_out = y_test_preprocess
print(img_out.shape)
index_out = np.array(label_test)

np.save('datasets/train_y', img_out)
np.save('datasets/label_y', index_out)
print('Write finished')
#
print(np.shape(np.load('datasets/train_y.npy')))
print(np.shape(np.load('datasets/label_y.npy')))

