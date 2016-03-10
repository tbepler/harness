import numpy as np
import sys
import random



#load sequences for language modeling
def split_data(gen, validation=0.05, labels=False, report=sys.stdout
               , xtype=np.int32, ytype=np.int32):
    data = list(gen) #data is list of (x,y) pairs
    if validation > 0:
        groups = [data]
        if labels: #split data by label
            groups = {}
            for x,y in data:
                d = groups.get(y, default=[])
                d.append((x,y))
                groups[y] = d
            groups = list(groups)
        training = []
        validation = []
        for group in groups:
            random.shuffle(group)
            m = int((1-validation)*len(group))
            training.extend(data[:m])
            validation.extend(data[m:])
    else:
        training = data
        validation = []
    print >>report, "Number of training sequences:", len(training)
    if len(validation) > 0:
        print >>report, "Number of validation sequences:", len(validation)
    #convert to matrix
    Xtrain = as_matrix(zip(*training)[0], xtype)
    Ytrain = as_matrix(zip(*training)[1], ytype)
    if len(validation) > 0:
        Xval = as_matrix(zip(*validation)[0], xtype)
        Yval = as_matrix(zip(*validation)[1], ytype)
    else:
        Xval = np.zeros((0,0), dtype=xtype)
        Yval = np.zeros((0,0), dtype=ytype)
    return Xtrain, Ytrain, Xval, Yval



def as_matrix(ss, dtype):
    m = max(len(s) for s in ss)
    n = len(ss)
    mat = np.full((m,n), -1, dtype=dtype) #fill with mask value
    for j in xrange(n):
        s = ss[j]
        mat[:len(s),j:j+1] = s
    return mat
    
