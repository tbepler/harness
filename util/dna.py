import numpy as np
import sys
import os

nucleotides = {'A':0, 'C':1, 'G':2, 'T':3, 'N':4}

class Label(object):
    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end
    def __str__(self):
        return ''.join([self.name,'(',str(self.start),':',str(self.end),')'])
    def __repr__(self):
        return self.__str__()

labeled = 0
unlabeled = 1
    
def load_data(paths, report=sys.stdout, **kwargs):
    import data
    dtypes = [parse_type(path) for path in paths]
    if not all(t == dtypes[0] for t in dtypes[1:]):
        print >>report, "Error: all files must be of same type but were {}".format(dtypes)
        raise Exception()
    dtype = dtypes[0]
    labels = dtype == labeled
    Xt, Yt, Xv, Yv = data.split_data(read_files(paths,report=report,**kwargs), labels=labels
                           , report=sys.stdout, **kwargs)
    if labels:
        labels = max(np.max(Yt), np.max(Yv))+1
    else:
        labels = 0
    return Xt, Yt, Xv, Yv, labels

def read_files(paths, report=sys.stdout, **kwargs):
    for gen in [read_file(path, report=report, **kwargs) for path in paths]:
        for x in gen:
            yield x
            
def read_file(path, report=sys.stdout, **kwargs):
    _,ext = os.path.splitext(path)
    if ext == '.fa' or ext == '.fna' or ext == '.fasta':
        return read_fasta(path, report=report, **kwargs)
    elif ext == '.lab':
        return read_labeled(path, report=report)
    raise Exception('Error: unrecognized extension {} on file {}'.format(ext,path))

def parse_type(path):
    _,ext = os.path.splitext(path)
    if ext == '.lab':
        return labeled
    return unlabeled

def read_labeled(path, report=sys.stdout):
    print >>report, "Reading file:", path
    with open(path) as f:
        for line in f:
            line = line.strip().upper()
            if len(line) > 0 and not line.startswith('#'):
                [x,y] = line.split()
                x = [nucleotides.get(b, -1) for b in x]
                y = int(y)
                yield x,y

def read_fasta(path, fragment=0, report=sys.stdout):
    seq = []
    print >>report, "Reading file:", path
    with open(path) as f:
        for line in f:
            if line[0] == '>':
                #this is a new sequence
                if len(seq) > 0:
                    if fragment > 1:
                        for i in xrange(0, len(seq), fragment):
                            end = min(i+fragment, len(seq))
                            if end-i >= 0.5*fragment:
                                yield seq[i:i+fragment-1], seq[i+1:i+fragment]
                    else:
                        yield seq[:-1], seq[1:]
                    seq = []
                name = line[1:-1]
                print >>report, "Loading sequence:", name
            else:
                for char in line.upper().strip():
                    if char in nucleotides:
                        seq.append(nucleotides[char])
                    else:
                        print 'Warning: unrecognized character \'{}\' detected in {}.'.format(
                            char, path)
        if len(seq) > 0:
            if fragment > 1:
                for i in xrange(0, len(seq), fragment):
                    end = min(i+fragment, len(seq))
                    if end-i >= 0.5*fragment:
                        yield seq[i:i+fragment-1], seq[i+1:i+fragment]
            else:
                yield seq[:-1], seq[1:]

