import argparse
import random
import os
import sys
import numpy as np

import harness.util.model
import harness.progress
import harness.data
import harness.savex

name = 'train'
description = 'Command for training models.'

def init_parser(parser):
    parser.add_argument('model', help='model file')
    parser.add_argument('files', metavar='FILE', nargs='+', help='input files')
    parser.add_argument('--max-iters', dest='max_iters', type=int, default=100, help='maximum number of iterations through the training data (default: 100)')
    parser.add_argument('--bptt', dest='bptt', type=int, default=None, help='the length at which BPTT should be truncated, 0 indicates full BPTT (default: model setting)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32, help='number of sequenes per batch (default: 32)')
    parser.add_argument('--fragment', dest='fragment', type=int, default=0, help='length into which input sequences should be fragmented, 0 indicates no fragmentation (default: 0)')
    parser.add_argument('--float', dest='dtype', choices=['32','64','default'], default='default', help='number of bits to use for floating point values')
    parser.add_argument('--validation', dest='validate', type=float, default=0.05, help='fraction of data to keep for validation (default: 0.05)')
    harness.save.save_parser(parser)

def run(args):
    #check whether specific float size was specified
    if args.dtype == '32':
        dtype = np.float32
    elif args.dtype == '64':
        dtype = np.float64
    else:
        dtype = None
    #load data
    #Xtrain, Ytrain, Xval, Yval, labels = genrnn.util.dna.load_data(args.files, report=sys.stdout
    #                                                               , fragment=args.fragment
    #                                                               , validation=args.validate)
    data, inputs, outputs = harness.data.load(*args.files)
    if args.validate > 0:
        n = int(args.validate*len(data))
        random.shuffle(data)
        val_data = data[:n]
        train_data = data[n:]
    else:
        val_data = None
        train_data = data
    
    model, model_name, epoch = harness.util.model.load_model(args.model, inputs, outputs, dtype=dtype)
    saver = harness.save.load_saver(args, model_name)
    
    print "Training", args.model
    train(model, train_data, val_data, args.max_iters
          , saver, start_epoch = epoch)
                
def train(model, train_data, val_data, max_iters, saver
          , start_epoch=0):

    steps = model.fit(train_data, max_iters=max_iters)
    err_, acc_, n_ = 0, 0, 0
    bar = harness.progress.progress_bar()
    next(bar)
    for iters, (err,acc,n) in steps:
        epoch = int(iters)+start_epoch+1
        h = 'Epoch {}/{}: [train]'.format(epoch, max_iters+start_epoch)
        p = iters % 1
        err_ += err
        acc_ += acc
        n_ += n
        bar.send((h, p, {'error':err_/n_, 'accuracy':float(acc_)/n_}))
        if p == 0:
            if val_data is not None:
                err_, acc_, n_ = 0, 0, 0
                h = 'Epoch {}/{}: [validate]'.format(epoch, max_iters+start_epoch)
                val_bar = harness.progress.progress_bar()
                next(val_bar)
                for iters, (err,acc,n) in model.validate(val_data):
                    err_ += err
                    acc_ += acc
                    n_ += n
                    val_bar.send((h, iters, {'error':err_/n_, 'accuracy':float(acc_)/n_}))
            saver.save(model, epoch, err_/n_, float(acc_)/n_, final=(int(iters)+1==max_iters))
            err_, acc_, n_ = 0, 0, 0



        
