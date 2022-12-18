import random
# Stefano Iannicelli
# this class move the complete dataset into the ram of gpu for optimize the time of training

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
