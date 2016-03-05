
import sys

def print_progress_bar(header, i, n, cols=40, stream=sys.stdout):
    p = float(i)/n
    fill = int(p*cols)
    bar = '#'*fill + ' '*(cols-fill)
    s = "\r\033[K{}  [{}] {:.1f}%".format(header, bar, p*100)
    print >>stream, s,
    stream.flush()
