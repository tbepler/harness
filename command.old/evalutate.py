import argparse
import numpy as np

import genrnn.util.model
import genrnn.util.dna
import genrnn.util.progress

name = 'evaluate'

parser = argparse.ArgumentParser("Command for evaluating models.")
parser.add_argument('model', help='model file')
parser.add_argument('files', metavar='FILE', nargs='+', help='input files')
parser.add_argument('--bptt', dest='bptt', type=int, default=2000, help='the length at which BPTT should be truncated, 0 indicates full BPTT (default: 128)')
parser.add_argument('--fragment', dest='fragment', type=int, default=0, help='length into which input sequences should be fragmented, 0 indicates no fragmentation (default: 0)')
parser.add_argument('--float', dest='dtype', choices=['32','64','default'], default='default', help='number of bits to use for floating point values')

def run(args):
    #check whether specific float size was specified
    if args.dtype == '32':
        dtype = np.float32
    elif args.dtype == '64':
        dtype = np.float64
    else:
        dtype = None
    inputs = len(util.dna.nucleotides)
    model,_,_ = util.model.load_model(args.model, inputs, inputs-1, dtype=dtype)
    data = util.dna.load_data(args.files, args.fragment)
    print "Evaluating", args.model
    evaluate(model, data, args.bptt)

def evaluate(model, data, bptt):
    m = len(data)
    for j in xrange(m):
        l,s = data[j]
        count = [0]
        X = s[:-1]
        Y = s[1:]
        def progress(k,n):
                count[0] += k
                if count[0] > 10000:
                    util.progress.print_progress_bar("Evaluating", j+float(k*X.shape[1])/n, m)
                    count[0] = count[0] % 10000
        err, acc = util.model.validate(model, X, Y, bptt, callback=progress)
        model.reset()
        print "\r\033[K",
        print "{}: error={}, accuracy={}".format(l,err, acc)




