import argparse
import random
import os
import sys
import numpy as np
import cPickle as pickle

import genrnn.util.model
import genrnn.util.dna
import genrnn.util.progress

name = 'train'
description = 'Command for training models.'

def init_parser(parser):
    parser.add_argument('model', help='model file')
    parser.add_argument('files', metavar='FILE', nargs='+', help='input files')
    parser.add_argument('--max-length', dest='max_len', type=int, default=None, help='maximum length at which input sequences should be truncated (default: No truncation)')
    parser.add_argument('--epochs', type=int, default=100, help='number of iterations through the training data (default: 100)')
    parser.add_argument('--bptt', dest='bptt', type=int, default=None, help='the length at which BPTT should be truncated, 0 indicates full BPTT (default: model setting)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32, help='number of sequenes per batch (default: 32)')
    parser.add_argument('--fragment', dest='fragment', type=int, default=0, help='length into which input sequences should be fragmented, 0 indicates no fragmentation (default: 0)')
    parser.add_argument('--float', dest='dtype', choices=['32','64','default'], default='default', help='number of bits to use for floating point values')
    parser.add_argument('--validation', dest='validate', type=float, default=0.05, help='fraction of data to keep for validation (default: 0.05)')
    save_parser(parser)

def run(args):
    #check whether specific float size was specified
    if args.dtype == '32':
        dtype = np.float32
    elif args.dtype == '64':
        dtype = np.float64
    else:
        dtype = None
    inputs = len(genrnn.util.dna.nucleotides)
    model, model_name, epoch = genrnn.util.model.load_model(args.model, inputs, inputs-1, dtype=dtype)
    saver = load_saver(args, model_name)
    #load data
    Xtrain, Ytrain, Xval, Yval = genrnn.util.dna.load_data(args.files, report=sys.stdout
                                                           , fragment=args.fragment
                                                           , validation=args.validate)
    print "Training", args.model
    train(model, Xtrain, Ytrain, Xval, Yval, args.epochs, args.bptt, args.batch_size
          , saver, start_epoch = epoch)
        
def as_matrix(data):
    ss = zip(*data)[1]
    m = max(len(s) for s in ss)
    n = len(ss)
    mat = np.full((m,n), -1, dtype=np.int32) #fill with mask value
    for j in xrange(n):
        s = ss[j]
        mat[:len(s),j:j+1] = s
    return mat
            
def train(model, Xtrain, Ytrain, Xval, Yval, epochs, bptt, batch_size, saver
          , start_epoch=0):
    start_epoch += 1
    epochs += start_epoch-1
    for epoch in xrange(start_epoch, epochs+1):
        h = "Epoch {}/{}:".format(epoch, epochs)
        #randomize the order of the training data
        perm = np.random.permutation(Xtrain.shape[1])
        Xtrain = Xtrain[:,perm]
        Ytrain = Ytrain[:,perm]
        #train
        count = [0]
        def progress(j,n):
            count[0] += j
            if count[0] > 1:
                genrnn.util.progress.print_progress_bar(h+" training", j, n)
                count[0] = count[0] % 1
        err,acc = genrnn.util.model.train(model, Xtrain, Ytrain, batch_size, bptt
                                          , callback=progress)
        print "\r\033[K{} [Training] error={}, accuracy={}".format(h, err, acc)
        sys.stdout.flush()
        #validate and reset
        if Xval.shape[1] > 0:
            count = [0]
            def progress(j,n):
                count[0] += j
                if count[0] > 1:
                    genrnn.util.progress.print_progress_bar(h+" validating", j, n)
                    count[0] = count[0] % 1
            err,acc = genrnn.util.model.validate(model, Xval, Yval, bptt
                                          , callback=progress)
            print "\r\033[K{} [Validation] error={}, accuracy={}".format(h, err, acc)
            sys.stdout.flush()
            model.reset()
        #check epoch and save model
        saver.save(model, epoch, err, acc, final=(epoch==epochs))

def save_parser(parser):
    parser.add_argument('--snapshot-method', dest='save_method', choices=['best','every'], default='best', help='method to use for saving models (default=best)')
    parser.add_argument('--snapshot-prefix', dest='save_prefix', default=None, help='path prefix where model snapshots should be saved (default: snapshots/model)')
    parser.add_argument('--snapshot-every', dest='snapshot_every', type=int, default=10, help='snapshot frequency in epochs, if snapshot-method=every (default: 10)')

def load_saver(args, model_name):
    if args.save_prefix is None:
        prefix = os.path.join('snapshots', model_name)
    else:
        prefix = args.save_prefix
    dr = os.path.dirname(prefix)
    if not os.path.exists(dr):
        os.makedirs(dr)
    save_format = ''.join([prefix, '_epoch{:06}_loss{:.3f}_acc{:.3f}.bin'])
    if args.save_method == 'best':
        return BestSaver(save_format)
    elif args.save_method == 'every':
        return EverySaver(save_format, args.snapshot_every)

class BestSaver(object):
    def __init__(self, save_format):
        self.save_format = save_format
        self.error = float('inf')

    def save(self, model, epoch, err, acc, final=False):
        if err < self.error:
            path = self.save_format.format(epoch, err, acc)
            with open(path, 'w') as f:
                pickle.dump(model, f)
            self.error = err

class EverySaver(object):
    def __init__(self, save_format, every):
        self.save_format = save_format
        self.every = every

    def save(self, model, epoch, err, acc, final=False):
        if final or epoch % self.every == 0:
            path = self.save_format.format(epoch, err, acc)
            with open(path, 'w') as f:
                pickle.dump(model, f)

        
