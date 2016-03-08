import numpy as np
import sys

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
