import numpy as np


w1 = np.load('weight_CFAR_B1.npy')
w2 = np.load('weight_CFAR_B2.npy')
w3 = np.load('weight_CFAR_B3.npy')
w4 = np.load('weight_CFAR_B4.npy')
w5 = np.load('weight_CFAR_B5.npy')
w6 = np.load('weight_CFAR_B6.npy')
w7 = np.load('weight_CFAR_B7.npy')
print('Done')
print(np.shape(w1))
print(np.shape(w2))
print(np.shape(w3))
print(np.shape(w4))
print(np.shape(w5))
print(np.shape(w6))
print(np.shape(w7))
a1 = np.shape(w1)
a2 = np.shape(w2)
a3 = np.shape(w3)
a4 = np.shape(w4)
a5 = np.shape(w5)
a6 = np.shape(w6)
a7 = np.shape(w7)
print(a1[0])
print(a1[1])

mean1 = np.sqrt(np.sum(w1**2)/(a1[0]*a1[1]))
mean2 = np.sqrt(np.sum(w2**2)/(a2[0]*a2[1]))
mean3 = np.sqrt(np.sum(w3**2)/(a3[0]*a3[1]))
mean4 = np.sqrt(np.sum(w4**2)/(a4[0]*a4[1]))
mean5 = np.sqrt(np.sum(w5**2)/(a5[0]*a5[1]))
mean6 = np.sqrt(np.sum(w6**2)/(a6[0]*a6[1]))
mean7 = np.sqrt(np.sum(w7**2)/(a7[0]*a7[1]))
print(mean1)
print(mean2)
print(mean3)
print(mean4)
print(mean5)
print(mean6)
print(mean7)

arr_m1 = np.mean(w1)
arr_m2 = np.mean(w2)
arr_m3 = np.mean(w3)
arr_m4 = np.mean(w4)
arr_m5 = np.mean(w5)
arr_m6 = np.mean(w6)
arr_m7 = np.mean(w7)
# print(arr_m1)

arr_std1 = np.std(w1)
arr_std2 = np.std(w2)
arr_std3 = np.std(w3)
arr_std4 = np.std(w4)
arr_std5 = np.std(w5)
arr_std6 = np.std(w6)
arr_std7 = np.std(w7)
# print(arr_std1)

noise1 = np.random.normal(arr_m1, arr_std1, size=(a1[0], a1[1])) / 50
noise2 = np.random.normal(arr_m2, arr_std2, size=(a2[0], a2[1])) / 50
noise3 = np.random.normal(arr_m3, arr_std3, size=(a3[0], a3[1])) / 50
noise4 = np.random.normal(arr_m4, arr_std4, size=(a4[0], a4[1])) / 50
noise5 = np.random.normal(arr_m5, arr_std5, size=(a5[0], a5[1])) / 50
noise6 = np.random.normal(arr_m6, arr_std6, size=(a6[0], a6[1])) / 50
noise7 = np.random.normal(arr_m7, arr_std7, size=(a7[0], a7[1])) / 50
mnoise1 = np.sqrt(np.sum(noise1**2)/(a1[0]*a1[1]))
mnoise2 = np.sqrt(np.sum(noise2**2)/(a2[0]*a2[1]))
mnoise3 = np.sqrt(np.sum(noise3**2)/(a3[0]*a3[1]))
mnoise4 = np.sqrt(np.sum(noise4**2)/(a4[0]*a4[1]))
mnoise5 = np.sqrt(np.sum(noise5**2)/(a5[0]*a5[1]))
mnoise6 = np.sqrt(np.sum(noise6**2)/(a6[0]*a6[1]))
mnoise7 = np.sqrt(np.sum(noise7**2)/(a7[0]*a7[1]))
print(mnoise1)
print(mnoise2)
print(mnoise3)
print(mnoise4)
print(mnoise5)
print(mnoise6)
print(mnoise7)

print(np.shape(noise1))
print(np.shape(noise2))
print(np.shape(noise3))
print(np.shape(noise4))
print(np.shape(noise5))
print(np.shape(noise6))
print(np.shape(noise7))

new_w1 = w1 + noise1
new_w2 = w2 + noise2
new_w3 = w3 + noise3
new_w4 = w4 + noise4
new_w5 = w5 + noise5
new_w6 = w6 + noise6
new_w7 = w7 + noise7


print(np.shape(new_w1))
print(np.shape(new_w2))
print(np.shape(new_w3))
print(np.shape(new_w4))
print(np.shape(new_w5))
print(np.shape(new_w6))
print(np.shape(new_w7))


np.save('weight_CFAR1', new_w1)
np.save('weight_CFAR2', new_w2)
np.save('weight_CFAR3', new_w3)
np.save('weight_CFAR4', new_w4)
np.save('weight_CFAR5', new_w5)
np.save('weight_CFAR6', new_w6)
np.save('weight_CFAR7', new_w7)
print('Write finished')
