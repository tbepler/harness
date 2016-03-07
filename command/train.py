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
    parser.add_argument('--bptt', dest='bptt', type=int, default=2000, help='the length at which BPTT should be truncated, 0 indicates full BPTT (default: 128)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32, help='number of sequenes per batch (default: 32)')
    parser.add_argument('--fragment', dest='fragment', type=int, default=0, help='length into which input sequences should be fragmented, 0 indicates no fragmentation (default: 0)')
    parser.add_argument('--float', dest='dtype', choices=['32','64','default'], default='default', help='number of bits to use for floating point values')
    parser.add_argument('--validate', dest='validate', default='on', choices=['on','off'], help='whether to partition data into training and validation sets when training (default: on)')
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
    data = genrnn.util.dna.load_data(args.files, args.fragment)
    print "Training", args.model
    dir = os.path.dirname(save_prefix)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if args.validate == 'off':
        validate = False
    else:
        validate = True
    train(model, data, args.epochs, args.bptt, args.batch_size, save_prefix
          , args.snapshot_every, start_epoch = epoch, do_validate=validate)
        
def as_matrix(data):
    ss = zip(*data)[1]
    m = max(len(s) for s in ss)
    n = len(ss)
    mat = np.full((m,n), -1, dtype=np.int32) #fill with mask value
    for j in xrange(n):
        s = ss[j]
        mat[:len(s),j:j+1] = s
    return mat
            
def train(model, data, epochs, bptt, batch_size, saver
          , start_epoch=0, do_validate=True):
    start_epoch += 1
    epochs += start_epoch-1
    if do_validate:
        #shuffle data
        random.shuffle(data)
        #hold out last 5% of sequences for validation
        m = int(0.95*len(data))
        training = data[:m]
        validation = data[m:]
    else:
        training = data
    print "Number of training sequences:", len(training)
    print "Training on:", sorted(zip(*training)[0])
    if do_validate:
        print "Number of validation sequences:", len(validation)
        print "Validating on:", sorted(zip(*validation)[0])
    #convert to np matrices
    training = as_matrix(training)
    if do_validate:
        validation = as_matrix(validation)
    for epoch in xrange(start_epoch, epochs+1):
        h = "Epoch {}/{}:".format(epoch, epochs)
        #randomize the order of the training data
        np.random.shuffle(training.T)
        #train
        count = [0]
        def progress(j,n):
            count[0] += j
            if count[0] > 1:
                genrnn.util.progress.print_progress_bar(h+" training", j, n)
                count[0] = count[0] % 1
        X = training[:-1]
        Y = training[1:]
        err,acc = genrnn.util.model.train(model, X, Y, batch_size, bptt, callback=progress)
        print "\r\033[K{} [Training] error={}, accuracy={}".format(h, err, acc)
        sys.stdout.flush()
        #validate and reset
        if do_validate:
            count = [0]
            def progress(j,n):
                count[0] += j
                if count[0] > 1:
                    genrnn.util.progress.print_progress_bar(h+" validating", j, n)
                    count[0] = count[0] % 1
            err,acc = genrnn.util.model.validate(model, validation[:-1,], validation[1:,], bptt
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

        
