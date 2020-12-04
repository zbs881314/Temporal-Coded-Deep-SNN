import numpy as np
import os
import matplotlib.pyplot as plt
import SNN

SAVE_PATH = os.getcwd() + '/weight_mnist'
mnist = SNN.MnistData(path=["MNIST/t10k-images.idx3-ubyte","MNIST/t10k-labels.idx1-ubyte"])

w1 = np.load(SAVE_PATH + '1.npy')
w2 = np.load(SAVE_PATH + '2.npy')

Ts = 1e-3
scale = 2
view_max = 2

l1 = SNN.SNNDiscrete(w1,Ts,scale)
l2 = SNN.SNNDiscrete(w2,Ts,scale)

xs, ys = mnist.next_batch(1, shuffle=True)
xs = (1-xs[0,:])/Ts
print(ys)

input_mat = np.zeros([784,int(1/Ts*view_max)])
input_mat[range(784),xs.astype(int)] = 1

l1out = l1.forward(input_mat)
l2out = l2.forward(l1out)

fg,ax = plt.subplots(nrows=2,ncols=1)
for i in range(np.shape(l2.potential)[0]):
    ax[1].plot(l2.potential[i])
    ax[0].plot(l2.current[i],label=str(i))
ax[0].legend(loc='upper left')
plt.show()

fg,ax = plt.subplots(nrows=1,ncols=2)
fg.set_size_inches(15,7)
for i in range(10):
    ax[0].plot(l2out[i],label=str(i))
ax[0].legend(loc='upper left')
ax[1].imshow(np.reshape(xs,[28,28]),cmap='gray')
plt.show()
