import argparse
import os
import numpy as np
from contextlib import contextmanager

import genrnn.util.model
import genrnn.util.dna
import genrnn.util.progress

name = 'annotate'
description = 'Command for annotating using models.'

def init_parser(parser):
    parser.add_argument('model', help='model file')
    parser.add_argument('files', metavar='FILE', nargs='+', help='input files')
    parser.add_argument('--bptt', dest='bptt', type=int, default=2000, help='the length at which BPTT should be truncated, 0 indicates full BPTT (default: 128)')
    parser.add_argument('--fragment', dest='fragment', type=int, default=0, help='length into which input sequences should be fragmented, 0 indicates no fragmentation (default: 0)')
    parser.add_argument('--float', dest='dtype', choices=['32','64','default'], default='default', help='number of bits to use for floating point values')
    parser.add_argument('--annotate-dest', dest='annotate_dest', default=None, help='destination to write annotations to (default: annotations/model_name)')
    parser.add_argument('--chrom-size', dest='chrom_size', default=None, help='path to chromosome sizes file for creating bigWig annotations')
    parser.add_argument('--annotate-url', dest='annotate_url', default='', help='url prefix for annotations')
    parser.add_argument('--bigwig-exe', dest='bigwig_exe', default='wigToBigWig', help='path to wigToBigWig executable (default: wigToBigWig)')

def run(args):
    #check whether specific float size was specified
    if args.dtype == '32':
        dtype = np.float32
    elif args.dtype == '64':
        dtype = np.float64
    else:
        dtype = None
    model, model_name, epoch = load_model(args.model, dtype=dtype)
    data = genrnn.util.dna.load_data(args.files, args.fragment)
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


@contextmanager
def bigwigs(model, dest, chrom_sizes, url, name, exe, email='anon@anon.com'):
    import rnn.dropout
    import genrnn.util.ucsc.hub
    import genrnn.util.ucsc.track
    genome = os.path.basename(chrom_sizes).split('.')[0]
    hub_path = os.path.join(dest, 'HUB')
    
    with genrnn.util.ucsc.hub.hub(hub_path, name=name, url_prefix=url) as hub:
        files = {}
        outputs = 0
        trackdb = hub[genome]
        trackdb[name] = genrnn.util.ucsc.track.SuperTrack(name)
        for i in xrange(len(model.layers)):
            l = model.layers[i]
            if hasattr(l, 'outputs'):
                outputs = l.outputs
            if not type(l) is rnn.dropout.Dropout: #ignore dropout layers
                layer_name = '-'.join([name, 'layer'+str(i)])
                trackdb[layer_name] = genrnn.util.ucsc.track.CompositeTrack(layer_name, parent=name
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
                    genrnn.util.progress.print_progress_bar("Annotating", j+float(k)/(n-1), m)
                    count[0] = count[0] % 10000
            for i in xrange(0, n-1, bptt):
                progress(i)
                Yh = model.predict(X[i:i+bptt])
                cent, cor, _ = genrnn.util.model.cross_ent_and_correct(Yh, Y[i:i+bptt])
                err += cent
                acc += cor
                #write annotations
                write_annotations(model, files)
            model.reset()
            err /= n
            acc /= float(n)
            print "\r\033[K",
            print "{}: error={}, accuracy={}".format(l,err, acc)
