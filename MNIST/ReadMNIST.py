import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("..")
import SNN

mnist = SNN.MnistData(path=["MNIST/train-images.idx3-ubyte","MNIST/train-labels.idx1-ubyte"])
xs,ys = mnist.next_batch(batch_size=10,shuffle=True)
for i in range(10):
    plt.imshow(np.reshape(xs[i],[28,28]))
    print(ys[i])
    plt.show()
