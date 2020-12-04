import numpy as np


w1 = np.load('weight_CFAR1.npy')
w2 = np.load('weight_CFAR2.npy')
w3 = np.load('weight_CFAR3.npy')
w4 = np.load('weight_CFAR4.npy')
w5 = np.load('weight_CFAR5.npy')
print('Done')
print(np.shape(w1))
print(np.shape(w2))
print(np.shape(w3))
print(np.shape(w4))
print(np.shape(w5))
a1 = np.shape(w1)
a2 = np.shape(w2)
a3 = np.shape(w3)
a4 = np.shape(w4)
a5 = np.shape(w5)
print(a1[0])
print(a1[1])

mean1 = np.sqrt(np.sum(w1**2)/(a1[0]*a1[1]))
mean2 = np.sqrt(np.sum(w2**2)/(a2[0]*a2[1]))
mean3 = np.sqrt(np.sum(w3**2)/(a3[0]*a3[1]))
mean4 = np.sqrt(np.sum(w4**2)/(a4[0]*a4[1]))
mean5 = np.sqrt(np.sum(w5**2)/(a5[0]*a5[1]))
print(mean1)
print(mean2)
print(mean3)
print(mean4)
print(mean5)

arr_m1 = np.mean(w1)
arr_m2 = np.mean(w2)
arr_m3 = np.mean(w3)
arr_m4 = np.mean(w4)
arr_m5 = np.mean(w5)
# print(arr_m1)

arr_std1 = np.std(w1)
arr_std2 = np.std(w2)
arr_std3 = np.std(w3)
arr_std4 = np.std(w4)
arr_std5 = np.std(w5)
# print(arr_std1)

noise1 = np.random.normal(arr_m1, arr_std1, size=(a1[0], a1[1])) / 50
noise2 = np.random.normal(arr_m2, arr_std2, size=(a2[0], a2[1])) / 50
noise3 = np.random.normal(arr_m3, arr_std3, size=(a3[0], a3[1])) / 50
noise4 = np.random.normal(arr_m4, arr_std4, size=(a4[0], a4[1])) / 50
noise5 = np.random.normal(arr_m5, arr_std5, size=(a5[0], a5[1])) / 50
mnoise1 = np.sqrt(np.sum(noise1**2)/(a1[0]*a1[1]))
mnoise2 = np.sqrt(np.sum(noise2**2)/(a2[0]*a2[1]))
mnoise3 = np.sqrt(np.sum(noise3**2)/(a3[0]*a3[1]))
mnoise4 = np.sqrt(np.sum(noise4**2)/(a4[0]*a4[1]))
mnoise5 = np.sqrt(np.sum(noise5**2)/(a5[0]*a5[1]))
print(mnoise1)
print(mnoise2)
print(mnoise3)
print(mnoise4)
print(mnoise5)

print(np.shape(noise1))
print(np.shape(noise2))
print(np.shape(noise3))
print(np.shape(noise4))
print(np.shape(noise5))

new_w1 = w1 + noise1
new_w2 = w2 + noise2
new_w3 = w3 + noise3
new_w4 = w4 + noise4
new_w5 = w5 + noise5

print(np.shape(new_w1))
print(np.shape(new_w2))
print(np.shape(new_w3))
print(np.shape(new_w4))
print(np.shape(new_w5))


np.save('weight_CFAR_F1', new_w1)
np.save('weight_CFAR_F2', new_w2)
np.save('weight_CFAR_F3', new_w3)
np.save('weight_CFAR_F4', new_w4)
np.save('weight_CFAR_F5', new_w5)
print('Write finished')
