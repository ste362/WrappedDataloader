# WrappedDataloader
An optimized way to move the dataset to gpu for improve train time
The module optimize.py contains the code class WrappedDataloader
The file CIFAR10_gpu_optimized.py is an example to how use the class WrappedDataloader

WrappedDataloader has three parameters:
  1. dataloader
  2. func is a function that preprocess and move the tensor to gpu
  3. shuffle (false by default)

This is an example of parameter func of WrappedDataloader for CIFAR10 dataset:
def preprocess(x, y):
    return x.view(-1, 3, 32, 32).to(device), y.to(device)  #device is the name of gpu
