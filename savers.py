import cPickle as pickle
import os

class Loader(object):
    def __init__(self):
        pass

    def argparse(self, parser):
        parser.add_argument('--save-method', dest='save_method', choices=['best','every'], default='best', help='method to use for saving models (default=best)')
        parser.add_argument('--save-prefix', dest='save_prefix', default=None, help='path prefix where model snapshots should be saved (default: snapshots/model)')
        parser.add_argument('--save-every', dest='save_every', type=int, default=1, help='save frequency in epochs, if save-method=every (default: 1)')

    def __call__(self, args, model_name):
        if args.save_prefix is None:
            prefix = os.path.join('snapshots', model_name)
        else:
            prefix = args.save_prefix
        dr = os.path.dirname(prefix)
        if not os.path.exists(dr):
            os.makedirs(dr)
        save_format = ''.join([prefix, '_epoch{:06}_loss{:.3f}.bin'])
        if args.save_method == 'best':
            return BestSaver(save_format)
        elif args.save_method == 'every':
            return EverySaver(save_format, args.snapshot_every)

class BestSaver(object):
    def __init__(self, save_format):
        self.save_format = save_format
        self.error = float('inf')

    def save(self, model, epoch, loss, final=False):
        if loss < self.error:
            path = self.save_format.format(epoch, loss)
            with open(path, 'w') as f:
                pickle.dump(model, f)
            self.error = loss

class EverySaver(object):
    def __init__(self, save_format, every):
        self.save_format = save_format
        self.every = every

    def save(self, model, epoch, loss, final=False):
        if final or epoch % self.every == 0:
            path = self.save_format.format(epoch, loss)
            with open(path, 'w') as f:
                pickle.dump(model, f)
