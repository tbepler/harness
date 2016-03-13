import numpy as np

import genrnn.util.model
import genrnn.util.dna
import genrnn.util.progress

name = 'bin'
description = 'Bin sequences by likelihood ratio with multiple models.'

def init_parser(parser):
    parser.add_argument('models', help='comma separated list of model files')
    parser.add_argument('files', metavar='FILE', nargs='+', help='input files')
    parser.add_argument('--bptt', dest='bptt', type=int, default=2000, help='the length at which BPTT should be truncated, 0 indicates full BPTT (default: 128)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32, help='number of sequenes per batch (default: 32)')
    parser.add_argument('--float', dest='dtype', choices=['32','64','default'], default='default', help='number of bits to use for floating point values')

def run(args):
    #check whether specific float size was specified
    if args.dtype == '32':
        dtype = np.float32
    elif args.dtype == '64':
        dtype = np.float64
    else:
        dtype = None
    inputs = len(genrnn.util.dna.nucleotides)
    models = [genrnn.util.model.load_model(path.strip(), inputs, inputs-1, dtype=dtype)
              for path in args.models.split(',')]
    models,names,_ = zip(*models)
    print "Binning", names
    n = 0
    correct = 0
    for path in args.files:
        if path == '-':
            (c_, n_) = bin(models, names, sys.stdin, args.bptt)
        else:
            with open(path) as f:
                (c_, n_) = bin(models, names, f, args.bptt)
        n += n_
        correct += c_
    print 'Accuracy:', correct/float(n)

def bin(models, names, f, bptt):
    n = 0
    correct = 0
    ps = np.zeros(len(models))
    for line in f:
        [seq,label] = line.split()
        x = encode(seq)
        ps[:] = 0
        for j in xrange(0, len(x), bptt):
	    xj = x[j:j+bptt,:]
            for i in xrange(len(models)):
                yh = models[i].predict(xj)
		for k in xrange(xj.shape[0]-1):
		    if 0 <= xj[k+1,0] < yh.shape[2]:
		        ps[i] += np.log(yh[k,0,xj[k+1,0]])
                #ps[i] += np.sum(np.log(yh[:,0,x]))
        for i in xrange(len(models)):
            models[i].reset()
        js = np.argwhere(ps == np.amax(ps))
        i = np.argmax(ps)
        n += 1
	#print seq, label, [names[j] for j in js], ps
        for j in js:
            if names[j] == label:
                correct += 1/float(len(js))
		#print 'Correct!'
	    else:
                print seq, label, [names[j] for j in js], ps
                print 'Incorrect'
    return correct, n

def encode(seq):
    x = []
    for c in seq:
        if c in genrnn.util.dna.nucleotides:
            x.append(genrnn.util.dna.nucleotides[c])
        else:
            x.append(-1)
            print 'Warning: character {} not recognized.'.format(c)
    x = np.array(x, dtype=np.int32)
    return x.reshape((len(x),1))
