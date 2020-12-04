import numpy as np
import os


class CifarData(object):
    def __init__(self, path):
        path[0] = os.getcwd() + "/" + path[0]
        path[1] = os.getcwd() + "/" + path[1]
        try:
            self.xs_full = np.load(path[0])
            self.ys_full = np.load(path[1])
            print(path[0] + ", " + path[1] + " " + "loaded")
        except:
            print("cannot load " + path[0] + ", " + path[1] + ", program will exit")
            exit(-1)
        self.datasize = np.shape(self.xs_full)[0]
        self.pointer = 0

    def next_batch(self, batch_size, shuffle=False):
        if shuffle:
            index = np.random.randint(self.datasize, size=batch_size)
            xs = self.xs_full[index, :]
            ys = self.ys_full[index, :]
            return xs, ys
        else:
            if self.pointer + batch_size < self.datasize:
                pass
            else:
                self.pointer = 0
                if batch_size >= self.datasize:
                    batch_size = self.datasize - 1
            xs = self.xs_full[self.pointer:self.pointer + batch_size, :]
            ys = self.ys_full[self.pointer:self.pointer + batch_size, :]
            self.pointer = self.pointer + batch_size
            return xs, ys

# TRAINING_BATCH = 20
# Kitti = CifarData(path=["KITTI/train_x.npy", "KITTI/label_x.npy"])
# for i in range(10):
#     xs, ys = Kitti.next_batch(TRAINING_BATCH, shuffle=True)
#     print(xs.shape)
#     print(ys.shape)
