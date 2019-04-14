from exp.nb_02 import *
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()

from torch import optim

class Dataset():
    def __init__(self, x, y): self.x,self.y = x,y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i],self.y[i]

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, **kwargs))