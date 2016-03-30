import sys
import data.parser

def lines(*args):
    for path in args:
        with open(path) as f:
            for line in f:
                yield line

def content(path):
    if path == '-':
        return sys.stdin.read()
    with open(path) as f:
        return f.read()    
                
def contents(*args):
    return ''.join([content(path) for path in args])
                
def load(*args):
    parser = data.parser.Parser()
    return parse(contents(*args))
