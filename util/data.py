import numpy as np
import sys
import random



#load sequences for language modeling
def split_data(gen, validation=0.05, labels=False, report=sys.stdout
               , xtype=np.int32, ytype=np.int32, **kwargs):
    data = list(gen) #data is list of (x,y) pairs
    if validation > 0:
        frac = validation
        groups = [data]
        if labels: #split data by label
            groups = {}
            for x,y in data:
                d = groups.get(y, [])
                d.append((x,y))
                groups[y] = d
            groups = groups.values()
        training = []
        validation = []
        for group in groups:
            random.shuffle(group)
            m = int((1-frac)*len(group))
            training.extend(group[:m])
            validation.extend(group[m:])
    else:
        training = data
        validation = []
    print >>report, "Number of training sequences:", len(training)
    if len(validation) > 0:
        print >>report, "Number of validation sequences:", len(validation)
    #convert to matrix
    Xtrain = as_matrix(zip(*training)[0], xtype)
    Ytrain = as_matrix(zip(*training)[1], ytype, labels=labels)
    if len(validation) > 0:
        Xval = as_matrix(zip(*validation)[0], xtype)
        Yval = as_matrix(zip(*validation)[1], ytype, labels=labels)
    else:
        Xval = np.zeros((0,0), dtype=xtype)
        Yval = np.zeros((0,0), dtype=ytype)
    return Xtrain, Ytrain, Xval, Yval



def as_matrix(ss, dtype, labels=False):
    m = 1
    if not labels:
        m = max(len(s) for s in ss)
    n = len(ss)
    mat = np.full((m,n), -1, dtype=dtype) #fill with mask value
    for j in xrange(n):
        s = ss[j]
        if labels:
            mat[0,j] = s
        else:
            mat[:len(s),j] = s
    return mat
    
