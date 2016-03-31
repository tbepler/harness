import numpy as np
import math
import random

class Minibatch(object):
    def __init__(self, data, size=256, randomize=True):
        self.data = data
        self.size = size
        self.randomize = randomize
        
    def __len__(self):
        return int(math.ceil(len(self.data)/float(self.size)))
    
    def __iter__(self):
        n = len(self.data)
        if self.randomize:
            p = np.random.permutation(n)
            for i in xrange(0, n, self.size):
                yield self.data[p[i:i+self.size]]
        else:
            for i in xrange(0, n, self.size):
                yield self.data[i:i+self.size]
            
class FromMatrix(object):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        if np.issubdtype(self.data.dtype, int):
            return self.data.shape[-1]
        return self.data.shape[-2]
    
    def __getitem__(self, i):
        return self.data[:-1,i], self.data[1:,i]

def from_matrix(data): return FromMatrix(data)
    
class FromList(object):
    def __init__(self, *args):
        self.data = args
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        try:
            ls = [[x[j] for j in i] for x in self.data]
        except Exception:
            ls = [x[i] for x in self.data]
        return [np.swapaxes(l, 0, 1) for l in ls if l.ndim > 1]

def from_list(*args): return FromList(*args)
    
def windows(xs, size, stride=1):
    for x in xs:
        for i in xrange(0, len(x)-size+1, stride):
            yield x[i:i+size]
