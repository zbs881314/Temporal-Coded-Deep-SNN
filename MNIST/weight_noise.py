import numpy as np


w1 = np.load('weight_scnn21.npy')
w2 = np.load('weight_scnn22.npy')
w3 = np.load('weight_scnn23.npy')
print('Done')
print(np.shape(w1))
print(np.shape(w2))
print(np.shape(w3))
a1 = np.shape(w1)
a2 = np.shape(w2)
a3 = np.shape(w3)
print(a1[0])
print(a1[1])

mean1 = np.sqrt(np.sum(w1**2)/(a1[0]*a1[1]))
mean2 = np.sqrt(np.sum(w2**2)/(a2[0]*a2[1]))
mean3 = np.sqrt(np.sum(w3**2)/(a3[0]*a3[1]))
print(mean1)
print(mean2)
print(mean3)

arr_m1 = np.mean(w1)
arr_m2 = np.mean(w2)
arr_m3 = np.mean(w3)
# print(arr_m1)

arr_std1 = np.std(w1)
arr_std2 = np.std(w2)
arr_std3 = np.std(w3)
# print(arr_std1)

noise1 = np.random.normal(arr_m1, arr_std1, size=(a1[0], a1[1])) / 50
noise2 = np.random.normal(arr_m2, arr_std2, size=(a2[0], a2[1])) / 50
noise3 = np.random.normal(arr_m3, arr_std3, size=(a3[0], a3[1])) / 50
mnoise1 = np.sqrt(np.sum(noise1**2)/(a1[0]*a1[1]))
mnoise2 = np.sqrt(np.sum(noise2**2)/(a2[0]*a2[1]))
mnoise3 = np.sqrt(np.sum(noise3**2)/(a3[0]*a3[1]))
print(mnoise1)
print(mnoise2)
print(mnoise3)

print(np.shape(noise1))
print(np.shape(noise2))
print(np.shape(noise3))

new_w1 = w1 + noise1
new_w2 = w2 + noise2
new_w3 = w3 + noise3

print(np.shape(new_w1))
print(np.shape(new_w2))
print(np.shape(new_w3))


np.save('weight_scnn211', new_w1)
np.save('weight_scnn221', new_w2)
np.save('weight_scnn231', new_w3)
print('Write finished')
