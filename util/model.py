import os
import imp
import cPickle as pickle

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
        end = min(n, j+batch_size)
        for i in xrange(0, m, bptt):
            if not callback is None:
                callback(j+(i/float(m))*(end-j), n)
            Yh = model.train(x[i:i+bptt,j:end], y[i:i+bptt,j:end])
            cent, cor, d = cross_ent_and_correct(Yh, y[i:i+bptt,j:end])
            err += cent
            acc += cor
            denom += d
        model.reset()
    return err/denom, acc/float(denom)

def validate(model, x, y, bptt, callback=None):
    (_,b) = x.shape
    err = 0
    acc = 0
    n = 0
    for i in xrange(0, x.shape[0], bptt):
        if not callback is None:
            callback(i,x.shape[0])
        X = x[i:i+bptt,]
        Y = y[i:i+bptt,]
        Yh = model.predict(X)
        cent, cor, m = cross_ent_and_correct(Yh, Y)
        err += cent
        acc += cor
        n += m
    return err/n, acc/float(n)
            
def cross_ent_and_correct( yh, y ):
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
                    assert False
                l = np.argmax(yh[i,j,])
                if k == l:
                    cor += 1
    return cent, cor, m




