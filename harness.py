import argparse
import imp
import cPickle as pickle
import numpy as np
import math
import os
import random
import sys

nucleotides = {'A':0, 'C':1, 'G':2, 'T':3}

parser = argparse.ArgumentParser("Harness for training and evaluating models.")
parser.add_argument('cmd', choices=['train','evaluate','annotate'], help='which command to run')
parser.add_argument('model', help='model file')
parser.add_argument('files', metavar='FILE', nargs='+', help='input files')
parser.add_argument('--max-length', dest='max_len', type=int, default=None, help='maximum length at which input sequences should be truncated (default: No truncation)')
parser.add_argument('--epochs', type=int, default=100, help='number of iterations through the training data (default: 100)')
parser.add_argument('--bptt', dest='bptt', type=int, default=2000, help='the length at which BPTT should be truncated, 0 indicates full BPTT (default: 128)')
parser.add_argument('--snapshot-prefix', dest='save_prefix', default=None, help='path prefix where model snapshots should be saved (default: snapshots/model)')
parser.add_argument('--snapshot-every', dest='snapshot_every', type=int, default=10, help='snapshot frequency in epochs (default: 10)')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=32, help='number of sequenes per batch (default: 32)')
parser.add_argument('--fragment', dest='fragment', type=int, default=0, help='length into which input sequences should be fragmented, 0 indicates no fragmentation (default: 0)')
parser.add_argument('--float', dest='dtype', choices=['32','64','default'], default='default', help='number of bits to use for floating point values')
parser.add_argument('--annotate-dest', dest='annotate_dest', default=None, help='destination to write annotations to (default: annotations/model_name)')
parser.add_argument('--chrom-size', dest='chrom_size', default=None, help='path to chromosome sizes file for creating bigWig annotations')
parser.add_argument('--annotate-url', dest='annotate_url', default='', help='url prefix for annotations')
parser.add_argument('--bigwig-exe', dest='bigwig_exe', default='wigToBigWig', help='path to wigToBigWig executable (default: wigToBigWig)')
parser.add_argument('--validate', dest='validate', default='on', choices=['on','off'], help='whether to partition data into training and validation sets when training (default: on)')

def main():
    args = parser.parse_args()
    model = load_model(args.model)
    #check whether specific float size was specified
    if args.dtype == '32':
        model.dtype = np.float32
    elif args.dtype == '64':
        model.dtype = np.float64
    #data = data_generator(args.files, args.max_len)
    model_name, epoch = model_name_epoch(args.model)
    if args.save_prefix is None:
        save_prefix = os.path.join('snapshots', model_name)
    else:
        save_prefix = args.save_prefix
    #load data
    data = load_data(args.files, args.fragment)
    #switch on command
    if args.cmd == 'train':
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
    elif args.cmd == 'evaluate': #evaluate
        print "Evaluating", args.model
        evaluate(model, data, args.bptt)
    else: #annotate
        if args.chrom_size is None:
            print "Error: chromosome sizes file must be specified for annotation."
            return
        print "Annotating", args.model
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        if args.annotate_dest is None:
            dest = os.path.join('annotations', model_name)
        else:
            dest = os.path.join(args.annotate_dest, model_name)
        url = os.path.join(args.annotate_url, model_name)
        annotate(model, data, args.bptt, dest, args.chrom_size, url, model_name, args.bigwig_exe)
    pass

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

def load_model(path, alphabet = len(nucleotides)):
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.py': #need to load model from python file
        model = imp.load_source("model", path)
        return model.build(alphabet)
    else: #unpickle model from binary file
        with open(path) as f:
            return pickle.load(f)

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
                    print_progress_bar("Evaluating", j+float(k*X.shape[1])/n, m)
                    count[0] = count[0] % 10000
        err, acc = validate(model, X, Y, bptt, callback=progress)
        model.reset()
        print "\r\033[K",
        print "{}: error={}, accuracy={}".format(l,err, acc)

from contextlib import contextmanager

@contextmanager
def bigwigs(model, dest, chrom_sizes, url, name, exe, email='anon@anon.com'):
    import rnn.dropout
    import util.ucsc.hub
    import util.ucsc.track
    genome = os.path.basename(chrom_sizes).split('.')[0]
    hub_path = os.path.join(dest, 'HUB')
    
    with util.ucsc.hub.hub(hub_path, name=name, url_prefix=url) as hub:
        files = {}
        outputs = 0
        trackdb = hub[genome]
        trackdb[name] = util.ucsc.track.SuperTrack(name)
        for i in xrange(len(model.layers)):
            l = model.layers[i]
            if hasattr(l, 'outputs'):
                outputs = l.outputs
            if not type(l) is rnn.dropout.Dropout: #ignore dropout layers
                layer_name = '-'.join([name, 'layer'+str(i)])
                trackdb[layer_name] = util.ucsc.track.CompositeTrack(layer_name, parent=name
                                                                     , type='bigWig')
                for j in xrange(outputs):
                    rpath = os.path.join('layer'+str(i), 'output'+str(j)+'.bw')
                    bwname = '-'.join([layer_name, 'output'+str(j)])
                    desc = ' '.join([name, 'Layer'+str(i), 'Output'+str(j)])
                    f = trackdb.bigwig(bwname, rpath, chrom_sizes, exe=exe, parent=layer_name)
                    files[(i,j)] = f
        yield files

def write_annotations(model, files):
    for (l,k),f in files.iteritems():
        Y = model.layers[l].Y[:,0,k]
        for i in xrange(Y.shape[0]):
            print >>f, Y[i]

def annotate(model, data, bptt, dest, chrom_sizes, url, name, exe):
    #open bigwig files
    with bigwigs(model, dest, chrom_sizes, url, name, exe) as files:
        m = len(data)
        for j in xrange(m):
            l,s = data[j]
            n = len(s)
            #update the bigwigs with the new sequence
            for f in files.itervalues():
                f.fixedStep(l.name, l.start+1)
            count = [0]
            X = s[:-1]
            Y = s[1:]
            err = 0
            acc = 0
            def progress(k):
                count[0] += k
                if count[0] > 10000:
                    print_progress_bar("Annotating", j+float(k)/(n-1), m)
                    count[0] = count[0] % 10000
            for i in xrange(0, n-1, bptt):
                progress(i)
                Yh = model.predict(X[i:i+bptt])
                cent, cor, _ = cross_ent_and_correct(Yh, Y[i:i+bptt])
                err += cent
                acc += cor
                #write annotations
                write_annotations(model, files)
            model.reset()
            err /= n
            acc /= float(n)
            print "\r\033[K",
            print "{}: error={}, accuracy={}".format(l,err, acc)
        
def as_matrix(data):
    ss = zip(*data)[1]
    m = max(len(s) for s in ss)
    n = len(ss)
    mat = np.full((m,n), -1, dtype=np.int32) #fill with mask value
    for j in xrange(n):
        s = ss[j]
        mat[:len(s),j:j+1] = s
    return mat
            
def train(model, data, epochs, bptt, batch_size, save_prefix, save_freq
          , start_epoch=0, do_validate=True):
    save_file_format = '{}_epoch{:06}_loss{:.3f}_acc{:.3f}.bin'
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
                print_progress_bar(h+" training", j, n)
                count[0] = count[0] % 1
        X = training[:-1]
        Y = training[1:]
        err,acc = train_epoch(model, X, Y, batch_size, bptt, callback=progress)
        print "\r\033[K{} [Training] error={}, accuracy={}".format(h, err, acc)
        sys.stdout.flush()
        #validate and reset
        if do_validate:
            count = [0]
            def progress(j,n):
                count[0] += j
                if count[0] > 1:
                    print_progress_bar(h+" validating", j, n)
                    count[0] = count[0] % 1
            err,acc = validate(model, validation[:-1,], validation[1:,], bptt, callback=progress)
            print "\r\033[K{} [Validation] error={}, accuracy={}".format(h, err, acc)
            sys.stdout.flush()
            model.reset()
        #check epoch and save model
        if epoch % save_freq == 0:
            save_path = save_file_format.format(save_prefix, epoch, err, acc)
            with open(save_path, 'w') as f:
                pickle.dump(model, f)
    save_path = '{}_final_loss{:.3f}_acc{:.3f}.bin'.format(save_prefix, err, acc)
    with open(save_path, 'w') as f:
        pickle.dump(model, f)

def print_progress_bar(header, i, n):
    p = float(i)/n
    cols = 40
    fill = int(p*cols)
    bar = '#'*fill + ' '*(cols-fill)
    s = "\r\033[K{}  [{}] {:.1f}%".format(header, bar, p*100)
    print s,
    sys.stdout.flush()

def train_sequence(model, x, y, bptt):
    for xb,yb in chunks(x, y, bptt):
        model.train(xb,yb)

def train_epoch(model, x, y, batch_size, bptt, callback=None):
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

#def train_batch(model, x, y, bptt, callback=None):
#    n = x.shape[0]
#    for i in xrange(0, n, bptt):
#        if not callback is None:
#            callback(i,n)
#        model.train(x[i:i+bptt,], y[i:i+bptt,])

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
            if k > -1:
                m += 1
                cent -= math.log(yh[i,j,k])
                l = np.argmax(yh[i,j,])
                if k == l:
                    cor += 1
    return cent, cor, m

def cross_entropy( yh, y ):
    (n,b) = y.shape
    err = 0
    m = 0
    for i in xrange(n):
        for j in xrange(b):
            k = y[i,j]
            if k > -1: #check for mask value in y
                err -= math.log(yh[i,j,k])
                m += 1
    return err, m

def partition(x, size):
    stride = len(x) if size == 0 else size
    return [x[i:i+stride] for i in xrange(0, len(x), stride)]

def chunks(x, y, size):
    stride = len(x) if size == 0 else size
    return ((x[i:i+stride],y[i:i+stride]) for i in xrange(0, len(x), stride))

class Label(object):
    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end
    def __str__(self):
        return ''.join([self.name,'(',str(self.start),':',str(self.end),')'])
    def __repr__(self):
        return self.__str__()

def load_data(paths, fragment_len, dtype=np.int32, verbose=True):
    data = []
    m = fragment_len
    for path in paths:
        if verbose: print "Reading file:", path
        with open(path) if path != '-' else sys.stdin as f:
            d = read_fasta(f, verbose=verbose)
            for k,s in d.iteritems():
                s = np.array(s, dtype=dtype)
                s = np.reshape(s, (len(s),1))
                n = len(s)
                if fragment_len > 0:
                    for i in xrange(0,n,fragment_len):
                        end = min(i+fragment_len, n)
                        #make sure fragment is at least half full
                        if end-i >= 0.5*fragment_len:
                            label = Label(k, i+1, end)
                            data.append((label,s[i:end]))
                else:
                    label = Label(k, 1, n)
                    data.append((label,s))
    return data

def read_fasta(f, verbose=True):
    seqs = {}
    name = ""
    seq = []
    for line in f:
        if line[0] == '>':
            #this is a new sequence
            name = line[1:-1]
            if verbose: print "Loading sequence:", name
            seq = [] #reset the sequence
            seqs[name] = seq
        else:
            for char in line.upper().strip():
                if char in nucleotides:
                    seq.append(nucleotides[char])
                else:
                    print 'Warning: unrecognized character \'{}\' detected in {}.'.format(char
                                                                                          , name)
                    seq.append(-1)
    return seqs

#def open_wig(path, 

if __name__=='__main__':
    main()
