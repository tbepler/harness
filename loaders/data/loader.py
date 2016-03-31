import sys
import parser as p

def init_parser(parser):
    parser.add_argument('--data-window', dest='data_window', type=int, default=0, help='window size in to which sequence data should be sliced, 0 or less means no windowing (default: 0)')
    parser.add_argument('--window-stride', dest='window_stride', type=int, default=1, help='stride to use when splitting sequences into windows (default: 1)')

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
                
def load(files, args):
    parser = p.Parser(window=args.data_window, stride=args.window_stride)
    return parser.parse(contents(*files))
