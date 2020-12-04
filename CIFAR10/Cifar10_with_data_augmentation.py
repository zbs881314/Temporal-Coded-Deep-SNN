import os
import numpy as np
import pickle
import tarfile
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
import random
import sys
try:
    # Python 3
    from urllib.request import urlretrieve
except:
    # Python 2
    from urllib import urlretrieve


def download_progress_hook(count, blocksize, totalsize):
    progress_size = int(count * blocksize)
    percent = min(100, int(count * blocksize * 100 / totalsize))
    sys.stdout.write("\r>>> %d%% (%d KB)" % (percent, progress_size / 1024))
    sys.stdout.flush()



def download_dataset(url, name, dir):
    filepath = os.path.join(dir, name)
    if not os.path.exists(filepath):
        if not os.path.exists(dir):
            os.mkdir(dir)
        print('Downloading '+ url + name + ' ...')
        filename, _ = urlretrieve(url + name, filepath, download_progress_hook)
        print(' >>> Done !')
    else:
        print(name + ' already downloaded ...')


def image_crop(images, shape):
    new_images = []
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        old_image = np.pad(old_image, [[1, 1], [1, 1], [0, 0]], 'constant')
        left = np.random.randint(old_image.shape[0] - shape[0] + 1)
        top = np.random.randint(old_image.shape[1] - shape[1] + 1)
        new_image = old_image[left:left + shape[0], top:top + shape[1], :]
        new_images.append(new_image)

    return np.array(new_images)


def image_crop_test(images, shape):
    new_images = []
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        old_image = np.pad(old_image, [[1, 1], [1, 1], [0, 0]], 'constant')
        left = int((old_image.shape[0] - shape[0]) / 2)
        top = int((old_image.shape[1] - shape[1]) / 2)
        new_image = old_image[left:left+shape[0], top:top+shape[1], :]
        new_images.append(new_image)
    return np.array(new_images)


def image_flip(images):
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        if np.random.random() < 0.5:
            new_image = cv2.flip(old_image, 1)
        else:
            new_image = old_image
        images[i, :, :, :] = new_image
    return images


def image_whiten(images):
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        old_image = old_image / 255
        mean = np.mean(old_image)
        std = np.std(old_image)
        new_image = (old_image - mean) / std
        new_image = new_image * 255
        images[i, :, :, :] = new_image
    return images


def image_noise(images, mean=0, std=0.01):
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        new_image = old_image
        for p in range(new_image.shape[0]):
            for j in range(new_image.shape[1]):
                for k in range(new_image.shape[2]):
                    new_image[p, j, k] += random.gauss(mean, std)
        images[i, :, :, :] = new_image
    return images


def data_augmentation(images, flip=False, crop=False, crop_shape=(24, 24, 3), whiten=False, noise=False, noise_mean=0, noise_std=0.01, mode='train'):
    if crop:
        if mode == 'train':
            images = image_crop(images, shape=crop_shape)
        elif mode == 'test':
            images = image_crop_test(images, shape=crop_shape)
    if flip:
        images = image_flip(images)
    if whiten:
        images = image_whiten(images)
    if noise:
        images = image_noise(images, mean=noise_mean, std=noise_std)
    return images


class CFAR10:
    def __init__(self):
        self.url  = 'http://www.cs.utoronto.ca/~kriz/'
        self.name = 'cifar-10-python.tar.gz'
        self.working_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../out/")

        # Fetch the data set
        download_dataset(self.url, self.name, self.working_dir)

        # Extract the files
        tar = tarfile.open(os.path.join(self.working_dir, self.name), "r:gz")
        tar.extractall(self.working_dir)
        datafiles = [s for s in tar.getnames() if 'data_batch' in s]
        testfiles  = [s for s in tar.getnames() if 'test_batch' in s]
        tar.close()
        datafiles.sort()

        # Extract the information
        x_concatenate = np.zeros(shape=(50000, 3072), dtype=np.uint8)
        y_concatenate = np.zeros(shape=(50000), dtype=np.uint8)
        offset = 0
        for datafile in datafiles:
            with open(os.path.join(self.working_dir, datafile), 'rb') as file:
                if sys.version_info > (3, 0):
                    train =  pickle.load(file, encoding='latin1')
                else:
                    train = pickle.load(file)
            x_concatenate[offset:offset+len(train['data'])] = train['data']
            y_concatenate[offset:offset+len(train['labels'])] = train['labels']
            offset += 10000

        for testfile in testfiles:
            with open(os.path.join(self.working_dir, testfile), 'rb') as file:
                if sys.version_info > (3, 0):
                    test =  pickle.load(file, encoding='latin1')
                else:
                    test = pickle.load(file)
            x = test['data']
            y = test['labels']

        # Reshape the training set
        train_ar = np.reshape(np.array(x_concatenate), (50000, 3, 32, 32))
        self.train_set = np.transpose(train_ar, (0,2,3,1))
        self.train_set = self.train_set
        self.train_label1 = np.array(y_concatenate)
        self.train_label = to_categorical(self.train_label1, 10)
        self.datasize = np.shape(self.train_set)[0]

        # Reshape the test set
        test_ar = np.reshape(np.array(x), (10000, 3, 32, 32))
        self.test_set = np.transpose(test_ar, (0,2,3,1))
        self.test_set = self.test_set
        self.test_label = np.array(y)
        self.test_label = to_categorical(self.test_label, 10)
        self.datasize_1 = np.shape(self.test_set)[0]
        self.pointer = 0

    def next_batch(self, batch_size, shuffle=False):
        if shuffle:
            index = np.random.randint(self.datasize, size=batch_size)
            xs_1 = self.train_set[index, :]
            ys_1 = self.train_label[index, :]
            # xs_new = image_noise(xs_1, mean=0, std=0.01)
            xs_new = data_augmentation(xs_1, flip=True, crop=True, crop_shape=(32, 32, 3), whiten=True, noise=False, noise_mean=0, noise_std=0.01, mode='train')
            xs_new = xs_new / 255
            return xs_new, ys_1
        else:
            if self.pointer + batch_size < self.datasize:
                pass
            else:
                self.pointer = 0
                if batch_size >= self.datasize:
                    batch_size = self.datasize
            xs_1 = self.train_set[self.pointer:self.pointer + batch_size, :]
            xs_new = data_augmentation(xs_1, flip=True, crop=True, crop_shape=(32, 32, 3), whiten=True, noise=False,
                                       noise_mean=0, noise_std=0.01, mode='train')
            xs_new = xs_new / 255
            ys_1 = self.train_label[self.pointer:self.pointer + batch_size, :]
            # ys_11 = self.train_label1[self.pointer:self.pointer + batch_size]
            self.pointer = self.pointer + batch_size
            return xs_new, ys_1


class_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
cfar10 = CFAR10()
print('Using ' + cfar10.url + cfar10.name + '...')
