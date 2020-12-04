import numpy as np


w1 = np.load('weight_CFAR_A1.npy')
w2 = np.load('weight_CFAR_A2.npy')
w3 = np.load('weight_CFAR_A3.npy')
w4 = np.load('weight_CFAR_A4.npy')
w5 = np.load('weight_CFAR_A5.npy')
w6 = np.load('weight_CFAR_A6.npy')
w7 = np.load('weight_CFAR_A7.npy')
w8 = np.load('weight_CFAR_A8.npy')
w9 = np.load('weight_CFAR_A9.npy')
w10 = np.load('weight_CFAR_A10.npy')
w11 = np.load('weight_CFAR_A11.npy')
w12 = np.load('weight_CFAR_A12.npy')
w13 = np.load('weight_CFAR_A13.npy')
w14 = np.load('weight_CFAR_A14.npy')
w15 = np.load('weight_CFAR_A15.npy')
w16 = np.load('weight_CFAR_A16.npy')
w17 = np.load('weight_CFAR_A17.npy')
print('Done')

a1 = np.shape(w1)
a2 = np.shape(w2)
a3 = np.shape(w3)
a4 = np.shape(w4)
a5 = np.shape(w5)
a6 = np.shape(w6)
a7 = np.shape(w7)
a8 = np.shape(w8)
a9 = np.shape(w9)
a10 = np.shape(w10)
a11 = np.shape(w11)
a12 = np.shape(w12)
a13 = np.shape(w13)
a14 = np.shape(w14)
a15 = np.shape(w15)
a16 = np.shape(w16)
a17 = np.shape(w17)


arr_m1 = np.mean(w1)
arr_m2 = np.mean(w2)
arr_m3 = np.mean(w3)
arr_m4 = np.mean(w4)
arr_m5 = np.mean(w5)
arr_m6 = np.mean(w6)
arr_m7 = np.mean(w7)
arr_m8 = np.mean(w8)
arr_m9 = np.mean(w9)
arr_m10 = np.mean(w10)
arr_m11 = np.mean(w11)
arr_m12 = np.mean(w12)
arr_m13 = np.mean(w13)
arr_m14 = np.mean(w14)
arr_m15 = np.mean(w15)
arr_m16 = np.mean(w16)
arr_m17 = np.mean(w17)
# print(arr_m1)

arr_std1 = np.std(w1)
arr_std2 = np.std(w2)
arr_std3 = np.std(w3)
arr_std4 = np.std(w4)
arr_std5 = np.std(w5)
arr_std6 = np.std(w6)
arr_std7 = np.std(w7)
arr_std8 = np.std(w8)
arr_std9 = np.std(w9)
arr_std10 = np.std(w10)
arr_std11 = np.std(w11)
arr_std12 = np.std(w12)
arr_std13 = np.std(w13)
arr_std14 = np.std(w14)
arr_std15 = np.std(w15)
arr_std16 = np.std(w16)
arr_std17 = np.std(w17)
# print(arr_std1)

noise1 = np.random.normal(arr_m1, arr_std1, size=(a1[0], a1[1])) / 50
noise2 = np.random.normal(arr_m2, arr_std2, size=(a2[0], a2[1])) / 50
noise3 = np.random.normal(arr_m3, arr_std3, size=(a3[0], a3[1])) / 50
noise4 = np.random.normal(arr_m4, arr_std4, size=(a4[0], a4[1])) / 50
noise5 = np.random.normal(arr_m5, arr_std5, size=(a5[0], a5[1])) / 50
noise6 = np.random.normal(arr_m6, arr_std6, size=(a6[0], a6[1])) / 50
noise7 = np.random.normal(arr_m7, arr_std7, size=(a7[0], a7[1])) / 50
noise8 = np.random.normal(arr_m8, arr_std8, size=(a8[0], a8[1])) / 50
noise9 = np.random.normal(arr_m9, arr_std9, size=(a9[0], a9[1])) / 50
noise10 = np.random.normal(arr_m10, arr_std10, size=(a10[0], a10[1])) / 50
noise11 = np.random.normal(arr_m11, arr_std11, size=(a11[0], a11[1])) / 50
noise12 = np.random.normal(arr_m12, arr_std12, size=(a12[0], a12[1])) / 50
noise13 = np.random.normal(arr_m13, arr_std13, size=(a13[0], a13[1])) / 50
noise14 = np.random.normal(arr_m14, arr_std14, size=(a14[0], a14[1])) / 50
noise15 = np.random.normal(arr_m15, arr_std15, size=(a15[0], a15[1])) / 50
noise16 = np.random.normal(arr_m16, arr_std16, size=(a16[0], a16[1])) / 50
noise17 = np.random.normal(arr_m17, arr_std17, size=(a17[0], a17[1])) / 50


new_w1 = w1 + noise1
new_w2 = w2 + noise2
new_w3 = w3 + noise3
new_w4 = w4 + noise4
new_w5 = w5 + noise5
new_w6 = w6 + noise6
new_w7 = w7 + noise7
new_w8 = w8 + noise8
new_w9 = w9 + noise9
new_w10 = w10 + noise10
new_w11 = w11 + noise11
new_w12 = w12 + noise12
new_w13 = w13 + noise13
new_w14 = w14 + noise14
new_w15 = w15 + noise15
new_w16 = w16 + noise16
new_w17 = w17 + noise17


print(np.shape(new_w1))
print(np.shape(new_w2))
print(np.shape(new_w3))
print(np.shape(new_w4))
print(np.shape(new_w5))
print(np.shape(new_w6))
print(np.shape(new_w7))
print(np.shape(new_w8))
print(np.shape(new_w9))
print(np.shape(new_w10))


np.save('weight_IMGN1', new_w1)
np.save('weight_IMGN2', new_w2)
np.save('weight_IMGN3', new_w3)
np.save('weight_IMGN4', new_w4)
np.save('weight_IMGN5', new_w5)
np.save('weight_IMGN6', new_w6)
np.save('weight_IMGN7', new_w7)
np.save('weight_IMGN8', new_w8)
np.save('weight_IMGN9', new_w9)
np.save('weight_IMGN10', new_w10)
np.save('weight_IMGN11', new_w11)
np.save('weight_IMGN12', new_w12)
np.save('weight_IMGN13', new_w13)
np.save('weight_IMGN14', new_w14)
np.save('weight_IMGN15', new_w15)
np.save('weight_IMGN16', new_w16)
np.save('weight_IMGN17', new_w17)
print('Write finished')
