import os
import numpy as np
import pickle
import tarfile
from keras.utils import to_categorical
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
    # TODO Deal with partial downloads
    # Check if data set was previously downloaded
    filepath = os.path.join(dir, name)
    if not os.path.exists(filepath):
        # Create the working directory for download if not already done
        if not os.path.exists(dir):
            os.mkdir(dir)
        print('Downloading '+ url + name + ' ...')
        filename, _ = urlretrieve(url + name, filepath, download_progress_hook)
        print(' >>> Done !')
    else:
        print(name + ' already downloaded ...')



class CFAR10:
    def __init__(self):
        self.url  = 'http://www.cs.utoronto.ca/~kriz/'
        self.name = 'cifar-10-python.tar.gz'
        self.working_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "out/")

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
        self.train_set = self.train_set / 255
        self.train_label = np.array(y_concatenate)
        self.train_label = to_categorical(self.train_label, 10)


        # Reshape the test set
        test_ar = np.reshape(np.array(x), (10000, 3, 32, 32))
        self.test_set = np.transpose(test_ar, (0,2,3,1))
        self.test_set = self.test_set / 255
        self.test_label = np.array(y)
        self.test_label = to_categorical(self.test_label, 10)
        self.datasize = np.shape(self.test_set)[0]
        self.pointer = 0


    def next_batch(self, batch_size, shuffle=False):
        if shuffle:
            index = np.random.randint(self.datasize, size=batch_size)
            xs = self.test_set[index, :]
            ys = self.test_label[index, :]
            return xs, ys
        else:
            if self.pointer + batch_size < self.datasize:
                pass
            else:
                self.pointer = 0
                if batch_size >= self.datasize:
                    batch_size = self.datasize - 1
            xs = self.test_set[self.pointer:self.pointer + batch_size, :]
            ys = self.test_label[self.pointer:self.pointer + batch_size, :]
            self.pointer = self.pointer + batch_size
            return xs, ys

# cfar10 = CFAR10()
# print ('Using ' + cfar10.url + cfar10.name + '...')
# #
# for i in range(10):
#     batch_size = 10
#     xs, ys = cfar10.next_batch(batch_size, shuffle=True)
#     print(xs.shape)
#     print(ys.shape)
