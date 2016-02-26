from collections import Counter
import argparse
import sys

parser = argparse.ArgumentParser('Script for building and evaluating ngram models.')
parser.add_argument('files', metavar='FILE', nargs='+', help='input files')
parser.add_argument('-n', dest='n', type=int, default=3, help='ngram size (default=3)')

def main():
    args = parser.parse_args()
    paths = args.files
    seqs = [read_sequence(path) for path in paths]
    training = seqs[:-1]
    print 'Training: ', paths[:-1]
    validation = seqs[-1]
    print 'Validation: ', paths[-1]
    for n in range(1,args.n+1):
        m = build_ngram_model(training, n)
        err,acc = evaluate(m, n, validation)
        print 'n={}, error={}, accuracy={}'.format(n, err, acc)
        sys.stdout.flush()


def evaluate(model, n, x):
    import math
    cent = 0
    acc = 0
    count = 0
    for i in xrange(len(x)-n+1):
        k = x[i:i+n-1]
        v = x[i+n-1]
        probs = model.get(k, Counter({'A':0.25, 'C':0.25, 'G':0.25, 'T':0.25}))
        amax = argmax(probs)
        if amax == v:
            acc += 1
        p = probs[v]
        cent -= math.log(p)
        count += 1
    return cent/float(count), acc/float(count)
        
def argmax(counter):
    ma = float('-inf')
    arg = None
    for k,v in counter.iteritems():
        if v > ma:
            arg = k
            ma = v
    return arg

def build_ngram_model(xs, n):
    m = {}
    for x in xs:
        ngram_counts(x, n, m)
    to_probabilities(m)
    return m

def build_ngram_models(xs, ns):
    models = []
    for n in ns:
        m = {}
        for x in xs:
            ngram_counts(x, n, m)
        to_probabilities(m)
        models.append(m)
    return models

def ngram_counts(x, n, counter):
    for i in xrange(len(x)-n+1):
        k = x[i:i+n-1]
        v = x[i+n-1]
        if k not in counter:
            counter[k] = Counter('ACGT')
        counter[k][v] += 1

def to_probabilities(ngram):
    for k,v in ngram.iteritems():
        s = sum(v.values())
        for k_ in v:
            v[k_] /= float(s)

def read_sequence(path):
    with open(path) as f:
        next(f) #discard header
        return ''.join([line.strip().upper() for line in f])

if __name__=='__main__':
    main()
