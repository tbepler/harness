import os
from contextlib import contextmanager

@contextmanager
def hub(path, **kwargs):
    h = Hub(path, **kwargs)
    try:
        yield h
    finally:
        h.close()

class Hub(object):

    def __init__(self, path, name=None, shortLabel=None, longLabel=None, genomesFile=None
                 , email=None, descriptionUrl=None, url_prefix=''):
        self._path = path
        self.hub = name
        self.shortLabel = shortLabel
        self.longLabel = longLabel
        self.genomesFile = genomesFile
        self.email = email
        self.descriptionUrl = descriptionUrl
        #check the hub path and load fields from it if it exists
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 0:
                        [key,value] = line.split()
                        if key[0] != '_' and key not in dir(self) or getattr(self,key) is None:
                            setattr(self, key, value)    
        #fill none fields with default values
        if self.hub is None:
            self.hub = 'defaultHub'
        if self.shortLabel is None:
            self.shortLabel = self.hub
        if self.longLabel is None:
            self.longLabel = self.hub
        if self.genomesFile is None:
            self.genomesFile = 'GENOMES'
        if self.email is None:
            self.email = 'anon@anon.com'
        if self.descriptionUrl is None:
            self.descriptionUrl = 'index.html'
        #open genomes
        self._genomes = Genomes(os.path.join(os.path.dirname(path), self.genomesFile), url_prefix)

    def close(self):
        #close genomes
        self._genomes.close()
        #write out hub file and close it
        with open(self._path, 'w') as f:
            print >>f, 'hub', self.hub
            for k in dir(self):
                v = getattr(self, k)
                if k[0] != '_' and k != 'hub' and not callable(v):
                    print >>f, k, v

    def __getitem__(self, x): return self._genomes[x]
    def __delitem__(self, x): del self._genomes[x]

                    
class Genomes(object):

    def __init__(self, path, url_prefix):
        self.path = path
        self.url_prefix = url_prefix
        self.genomes = {}
        if os.path.exists(path):
            dirname = os.path.dirname(path)
            with open(path) as f:
                lines = [l.strip() for l in f.readlines() if len(l.strip()) > 0]
                for i in xrange(0, len(lines), 2):
                    genome = lines[i].split()[1]
                    trackdb = lines[i+1].split()[1]
                    trackdb_path = os.path.join(dirname, trackdb)
                    trackdb_url = os.path.join(url_prefix, os.path.dirname(trackdb_path))
                    trackdb = TrackDB(trackdb_path, trackdb_url)

    def close(self):
        dirname = os.path.dirname(self.path)
        with open(self.path, 'w') as f:
            for genome,trackdb in self.genomes.iteritems():
                #close this trackdb
                trackdb.close()
                print >>f, 'genome', genome
                print >>f, 'trackDb', os.path.relpath(trackdb.path, dirname) #need relative path
                print >>f, ''

    def __getitem__(self, x):
        if x in self.genomes:
            return self.genomes[x]
        dirname = os.path.join(os.path.dirname(self.path), x)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        trackdb_path = os.path.join(dirname, 'TRACKS')
        trackdb_url = os.path.join(self.url_prefix, x)
        trackdb = TrackDB(trackdb_path, trackdb_url)
        self.genomes[x] = trackdb
        return trackdb

    def __delitem__(self, x): del self.genomes[x]
                
                    
class TrackDB(object):

    def __init__(self, path, url_prefix):
        self.path = path
        self.url_prefix = url_prefix
        self.tracks = {}
        if os.path.exists(path):
            with open(path) as f:
                name = ''
                track = NullTrack()
                for line in f:
                    line = line.strip()
                    if len(line) > 0:
                        tokens = line.split()
                        k = tokens[0]
                        v = ' '.join(tokens[1:])
                        if k == 'track':
                            if name != '':
                                self.tracks[name] = track
                                name = v
                                track = NullTrack()
                            else:
                                track[k] = v
                #add last track
                if name != '':
                    self.tracks[name] = track
            
    def close(self):
        #on close, write trackdb file
        with open(self.path, 'w') as f:
            for name,track in self.tracks.iteritems():
                #close the track
                track.close()
                print >>f, 'track', name
                for k,v in track.metadata:
                    print >>f, k, v
                print >>f, ''

    def bigwig(self, name, relpath, chrom_sizes, shortLabel=None, longLabel=None, **kwargs):
        import bigwig
        if shortLabel is None:
            shortLabel = name
        if longLabel is None:
            longLabel = shortLabel
        abspath = os.path.join(os.path.dirname(self.path),relpath)
        dirname = os.path.dirname(abspath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        url = os.path.join(self.url_prefix, relpath)
        bw = bigwig.Bigwig(abspath, chrom_sizes, url=url, shortLabel=shortLabel
                           , longLabel=longLabel, **kwargs)
        self.tracks[name] = bw
        return bw

    def __getitem__(self, x): return self.tracks[x]
    def __setitem__(self, x, d): self.tracks[x] = d
    def __delitem__(self, x): del self.tracks[x]

class NullTrack(object):

    def __init__(self):
        self._metadata = {}

    def __getitem__(self, x): return self._metadata[x]
    def __setitem__(self, x, y): self._metadata[x] = y
    def __delitem__(self, x): del self._metadata[x]

    @property
    def metadata(self): return self._metadata.iteritems()

    def close(self):
        pass
