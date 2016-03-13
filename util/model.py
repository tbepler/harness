import os
import imp
import math
import cPickle as pickle
import numpy as np

def load_model(path, inputs, outputs, dtype = None):
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.py': #need to load model from python file
        model = imp.load_source("model", path)
        m = model.build(inputs, outputs)
    else: #unpickle model from binary file
        with open(path) as f:
            m = pickle.load(f)
    if dtype is not None:
        m.dtype = dtype
    name,epoch = model_name_epoch(path)
    return m, name, epoch

def model_name_epoch(path):
    base = os.path.splitext(os.path.basename(path))[0]
    splt = base.split('epoch')
    if len(splt) > 1:
        name = splt[0][:-1]
        epoch = int(splt[1].split('_')[0])
    else:
        name = splt[0]
        epoch = 0
    return name,epoch

def train(model, x, y, batch_size, bptt, callback=None):
    (m,n) = x.shape
    err = 0
    acc = 0
    denom = 0
    for j in xrange(0, n, batch_size):
        size = min(n-j, batch_size)
        start = 0
        end = 0
        for yh in model.train(x[:,j:j+batch_size], y[:,j:j+batch_size], bptt=bptt):
            end += yh.shape[0]
            cent, cor, d = cross_ent_and_correct(yh, y[start:end, j:j+batch_size])
            err += cent
            acc += cor
            denom += d
            start = end
            if not callback is None:
                callback(j+end*size/float(m), n)
    return err/denom, acc/float(denom)

def validate(model, x, y, bptt, callback=None):
    (k,b) = x.shape
    err = 0
    acc = 0
    n = 0
    start = 0
    end = 0
    for yh in model.predict(x, bptt=bptt):
        end += yh.shape[0]
        cent, cor, m = cross_ent_and_correct(yh, y[start:end])
        err += cent
        acc += cor
        n += m
        start = end
        if not callback is None:
            callback(end, k)
    return err/n, acc/float(n)
            
def cross_ent_and_correct( yh, y ):
    #print y
    (n,b) = y.shape
    cent = 0
    cor = 0
    m = 0
    for i in xrange(n):
        for j in xrange(b):
            k = y[i,j]
            if -1 < k < yh.shape[2]:
                m += 1
                try:
                    cent -= math.log(yh[i,j,k])
                except Exception as e:
                    print e
                    print i, j, k, yh[i,j,k]
                l = np.argmax(yh[i,j,])
                if k == l:
                    cor += 1
    return cent, cor, m




