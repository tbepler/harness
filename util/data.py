import numpy as np
import sys
import random

language = 'language'
classify = 'classify'
translate = 'translate'

#load sequences for language modeling
def load_data(gen, validation=0.05, report=sys.stdout, dtype=np.int32, method=language):
    data = list(gen) #data is list of (sequence,label) or (sequence,sequence) pairs
    # (sequence,label) for method='language' or method='classify'
    # (sequence,sequence) for method='translate'
    if validation > 0:
        random.shuffle(data)
        m = int((1-validation)*len(data))
        training = data[:m]
        validation = data[m:]
    else:
        training = data
        validation = []
    print >>report, "Number of training sequences:", len(training)
    if method == language:
        print >>report, "Training on:", sorted(zip(*training)[1])
    if len(validation) > 0:
        print >>report, "Number of validation sequences:", len(validation)
        if method == language:
            print >>report, "Validating on:", sorted(zip(*validation)[1])
    #convert to matrix
    training = as_matrix(zip(*training)[0], dtype)
    if len(validation) > 0:
        validation = as_matrix(validation, dtype)
    else:
        validation = np.zeros((0,0), dtype=dtype)
    Xtrain = training[:-1]
    Ytrain = training[1:]
    Xval = validation[:-1]
    Yval = validation[1:]
    return Xtrain, Ytrain, Xval, Yval

def as_matrix(ss, dtype):
    m = max(len(s) for s in ss)
    n = len(ss)
    mat = np.full((m,n), -1, dtype=dtype) #fill with mask value
    for j in xrange(n):
        s = ss[j]
        mat[:len(s),j:j+1] = s
    return mat
    
