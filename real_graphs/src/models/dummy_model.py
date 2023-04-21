import torch

'''Dummy model which returns tensor of zeros in forward pass'''

class Dummy_model():
    def __init__(self, n_classes, device = 0):
        self.out_dim = n_classes
        self.device = device

    def eval(self):
        pass

    def train(self): 
        pass

    def __call__(self, attr, adj):
        return torch.zeros(attr.shape[0], self.out_dim, device = self.device)