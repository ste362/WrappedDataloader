# WrappedDataloader
An optimized way to move the dataloader to gpu for improve train time.
The idea is to move the full dataloader to gpu ram for maximize the parallelism of gpu and don't lose time for move each batch fro the system ram to gpu ram.

The module optimize.py contains the class WrappedDataloader.

The file pytorch_gpu_optimized.py is an example to how use the class WrappedDataloader.

WrappedDataloader has three parameters:
  1. dataloader
  2. func is a function that preprocess and move the tensor to gpu
  3. shuffle (false by default)
```
class WrappedDataLoader:
    def __init__(self, dataloader, func, shuffle=False): #func is a function used of preprocess and move the tensor to gpu
        self.dataloader = dataloader
        self.func = func
        self.shuffle=shuffle
        self.address=[]
        batches = iter(self.dataloader)
        for b in batches:
            self.address.append(self.func(*b))

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.address)
            return iter(self.address)
        return iter(self.address)
```

This is an example of parameter func of WrappedDataloader for CIFAR10 dataset:
```
def preprocess(x, y):
    return x.view(-1, 3, 32, 32).to(device), y.to(device)  #device is the name of gpu
```
This is an example of parameter func of WrappedDataloader for FashionMNIST dataset:
```
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(device), y.to(device)
```
Example to how use class WrappedDataLoader
```
train_dataloader = WrappedDataLoader(train_dataloader, preprocess,shuffle=True)
test_dataloader = WrappedDataLoader(test_dataloader, preprocess,shuffle=True)
```
With this module is possible to achieve up to 3 of speedup during the train.

The following table shows the results obtained:

| Program             | Standard epoch time   | Optimized epoch time  |
| -------------       |:-------------:        | -----:                |
| Train FashionMNIST  |        ~13s           |           ~7s         |
| Train CIFAR-10      |        ~14s           |           ~6s         |

