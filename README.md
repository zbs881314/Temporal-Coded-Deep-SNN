# Temporal-Coded Deep Spiking Neural Networks with Easy Training and Robust Performance
if you use the code, please cite the paper"Temporal-Coded Deep Spiking Neural Network with Easy Training and Robust Performance", https://arxiv.org/pdf/1909.10837.pdf  
This package includes verification source code for the paper "Temporal-Coded Deep Spiking Neural Network with Easy Training and Robust Performance".  
   
The code was tested with Python 3.7, Tensorflow 1.13.1, in Ubuntu 18 with NVIDIA GeForce RTX 2080Ti GPU, and Intel QuadCore with 32GB computer memory.  
   
One can set "device_count={'GPU': 0}" to use CPU for testing if GPU memory is not large enough.  

The CIFAR10 medium/large model and ImageNet model need large GPU memory. They were trained and tested on multiple GPUs. The sample code has been adjusted to use very small batch size to fit in a single GPU. For code demonstration only, one can further reduce batch size or run on CPU to avoid GPU resouce exhaust error.   
   
=======================================================================================   
A quick way to verify some of the results presented in the paper is to enter the MNIST folder to run:  

               python3 TestSCNN.py  

or to enter the CIFAR10 folder to run:  

               python3 TestCifar_S.py  

The pre-trained weights will be loaded to reproduce the testing accuracy results of Table 2 (for MNIST) and of Supplementary Section A.2, Paragraph 2 (for CIFAR10 small model).     
You can also enter the ImageNet folder to run:  

               python3 TestImageNet.py  

However, because the pre-trained weights are too big to include in this package, only random weights will be used to generate random testing results. Similarly, pre-trained weights are not included for CIFAR10 medium and large models. So the code of   
TestCifar_L.py and TestCifar_M.py will have random outputs.   
  
========================================================================================   
   
The training procedure can be verified as follows. Enter the MNIST folder to run:  

                python3 TrainSCNN.py  
or  
                python3 trainMNIST.py  
  
or enter the CIFAR10 folder to run:  
  
                python3 TrainCifar_S.py  
or  
                python3 TrainCifar_M.py  
or  
                python3 TrainCifar_L.py  
  
==========================================================================================  
  
To verify the weight noise perturbation effect, enter each folder to run weight_noise*.py, which will load pre-trained weights, add noise, and save to new data file *.npy. Then revise the Test*.py to use these new weights to verify classification accuracy results of Fig. 2 of the paper.  
  
==========================================================================================  
A list of python files with brief explanation:  
  
Inside the MNIST folder:  
  
TestSCNN.py: load pre-trained weights, calculate test accuracy over testing dataset  
TrainSCNN.py: training program for SNN with two SCNN layers (included in the paper)  
trainMNIST.py: training program for SNN with two FC layers only (not included in the paper)  
Visualize.py: visualize the input/output neuron values as images  
weight_noise.py: add noise to weights, save weights to new npy data files to be read by TestSCNN.py  
  
Inside the CIFAR10 folder:  
  
TestCifar_L.py:  load pre-trained weights, calculate test accuracy over testing dataset  
TestCifar_M.py:  (for large, medium, small models)  
TestCifar_S.py:  
TrainCifar_L.py: training program for large, medium, and small models  
TrainCifar_M.py:  
TrainCifar_S.py:  
weight_noise_l.py: add noise to weights, save weights to new npy data files to be read by TestCifar_?.py  
weight_noise_m.py:  
weight_noise_s.py:  
  
Inside the ImageNet folder:  
  
TestImageNet.py: load pre-trained weights, calculate test accuracy over testing dataset  
weight_noise.py: add noise to weights, save weights to new npy data files to be read by TestImageNet.py  
  

