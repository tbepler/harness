import sys
import time
from contextlib import contextmanager

class Loader(object):
    def __init__(self):
        pass

    def argparse(self, argparse):
        argparse.add_argument('--verbose/-v', dest='verbose', type=int, default=2
                              , choices=[0,1,2], help='verbosity level (default: 2)')
        argparse.add_argument('--log-file', dest='log_file', default=None, help='file to write execution log to (default: stdout)')

    @contextmanager
    def __call__(self, args):
        if args.log_file is not None:
            with open(args.log_file, 'w') as f:
                yield self.make_logger(f, args.verbose)
        else:
            yield self.make_logger(sys.stdout, args.verbose)

    def make_logger(self, out, verbose):
        if verbose == 2:
            return ProgressLogger(out)
        elif verbose == 1:
            return BasicLogger(out)
        elif verbose == 0:
            return NullLogger()

class ProgressLogger(object):
    def __init__(self, out=sys.stdout, delta=1.0, bar_len=40):
        self.out = out
        self.delta = delta
        self.bar_len = bar_len
        self.clear_code = ''

    def write(self, x):
        self.out.write(self.clear_code)
        self.out.write(x)
        self.clear_code = ''

    def progress_monitor(self):
        def progress(p, label):
            if p == 0:
                progress.tstart = time.time()
                progress.t = progress.tstart
            tcur = time.time()
            if tcur - progress.t >= self.delta:
                progress.t = tcur
                #write a hash first to mark the start of line, but clear it
                self.out.write('#\r\033[K') 
                self.out.write(self.clear_code)
                n = int(p*self.bar_len)
                bar = ''.join(['#']*n + [' ']*(self.bar_len-n))
                if p == 0 or p == 1:
                    eta = 0
                else:
                    eta = (tcur-progress.tstart)/p*(1-p)
                hours, rem = divmod(eta, 3600)
                mins, secs = divmod(rem, 60)
                line = '# {} [{}] {:7.2%}, eta {:0>2}:{:0>2}:{:0>2}'.format(label, bar, p
                                                                            , int(hours)
                                                                            , int(mins)
                                                                            , int(secs))
                self.out.write(line)
                self.out.write('\n')
                self.out.flush()
                self.clear_code = '\033[1F\033[K'
        return progress

    def flush(self):
        self.out.flush()

class BasicLogger(object):
    def __init__(self, out):
        self.out = out

    def write(self, x):
        self.out.write(x)

    def progress_monitor(self):
        def progress(*args, **kwargs):
            pass
        return progress

    def flush(self):
        self.out.flush()
        
class NullLogger(object):
    def __init__(self):
        pass

    def write(self, x):
        pass

    def progress_monitor(self):
        def progress(*args, **kwargs):
            pass
        return progress

    def flush(self):
        pass

