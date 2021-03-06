
class SuperTrack(object):

    def __init__(self, group, shortLabel=None, longLabel=None, **kwargs):
        self.group = group
        self.shortLabel = group if shortLabel is None else shortLabel
        self.longLabel = shortLabel if longLabel is None else longLabel
        self.kwargs = kwargs

    @property
    def metadata(self):
        yield 'group', self.group
        yield 'shortLabel', self.shortLabel
        yield 'longLabel', self.longLabel
        yield 'superTrack', 'on'
        for k,v in kwargs.iteritems():
            yield k, v

    def __str__(self): return self.group

class CompositeTrack(object):

    def __init__(self, group, shortLabel=None, longLabel=None, **kwargs):
        self.group = group
        self.shortLabel = group if shortLabel is None else shortLabel
        self.longLabel = shortLabel if longLabel is None else longLabel
        self.kwargs = kwargs

    @property
    def metadata(self):
        yield 'group', self.group
        yield 'shortLabel', self.shortLabel
        yield 'longLabel', self.longLabel
        yield 'compositeTrack', 'on'
        for k,v in kwargs.iteritems():
            yield k, v

    def __str__(self): return self.group
