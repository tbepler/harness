import subprocess as sp
from contextlib import contextmanager

class Bigwig(object):

    def __init__(self, dest, chrom_sizes
                 , exe='wigToBigWig'
                 #, description="A bigWig file"
                 , url=""
                 , **kwargs):
        self.process = sp.Popen([exe, 'stdin', chrom_sizes, dest], stdin=sp.PIPE)
        #self.desc = description
        self.url = url
        self.kwargs = kwargs

    @property
    def handle(self):
        return self.process.stdin
        
    def track_fields(self):
        yield 'type=bigWig'
        #yield 'description=\"'+self.desc+'\"'
        yield 'bigDataUrl='+self.url
        for k,v in self.kwargs.iteritems():
            yield k+'='+v

    @property
    def metadata(self):
        yield 'type', 'bigWig'
        #yield 'description', '\"'+self.desc+'\"'
        yield 'bigDataUrl', self.url
        for k,v in self.kwargs.iteritems():
            yield k, v

    @property
    def track(self):
        return ' '.join(self.track_fields())

    def fixedStep(self, chrom, start=1, step=1, span=None):
        print >>self.handle, 'fixedStep chrom='+chrom+' start='+str(start)+' step='+str(step),
        if span is not None:
            print >>self.handle, 'span='+str(span)
        else:
            print >>self.handle, ''

    def write(self, x):
        self.handle.write(x)

    def close(self):
        self.process.stdin.close()
        self.process.wait()

@contextmanager
def bigwig(dest, chrom_size, **kwargs):
    h = Bigwig(dest, chrom_size, **kwargs)
    try:
        yield h
    finally:
        h.close()
        

