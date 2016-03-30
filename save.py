import cPickle as pickle
import argsparse
import os

def save_parser(parser):
    parser.add_argument('--snapshot-method', dest='save_method', choices=['best','every'], default='best', help='method to use for saving models (default=best)')
    parser.add_argument('--snapshot-prefix', dest='save_prefix', default=None, help='path prefix where model snapshots should be saved (default: snapshots/model)')
    parser.add_argument('--snapshot-every', dest='snapshot_every', type=int, default=10, help='snapshot frequency in epochs, if snapshot-method=every (default: 10)')

def load_saver(args, model_name):
    if args.save_prefix is None:
        prefix = os.path.join('snapshots', model_name)
    else:
        prefix = args.save_prefix
    dr = os.path.dirname(prefix)
    if not os.path.exists(dr):
        os.makedirs(dr)
    save_format = ''.join([prefix, '_epoch{:06}_loss{:.3f}_acc{:.3f}.bin'])
    if args.save_method == 'best':
        return BestSaver(save_format)
    elif args.save_method == 'every':
        return EverySaver(save_format, args.snapshot_every)

class BestSaver(object):
    def __init__(self, save_format):
        self.save_format = save_format
        self.error = float('inf')

    def save(self, model, epoch, err, acc, final=False):
        if err < self.error:
            path = self.save_format.format(epoch, err, acc)
            with open(path, 'w') as f:
                pickle.dump(model, f)
            self.error = err

class EverySaver(object):
    def __init__(self, save_format, every):
        self.save_format = save_format
        self.every = every

    def save(self, model, epoch, err, acc, final=False):
        if final or epoch % self.every == 0:
            path = self.save_format.format(epoch, err, acc)
            with open(path, 'w') as f:
                pickle.dump(model, f)
