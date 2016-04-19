import pandas as pd

import harness.model as m
import harness.logger as lg

class Activation(object):
    def __init__(self, parser, model_loader=m.Loader(), logger_loader=lg.Loader()):
        self.name = 'activation'
        self.description = 'Command for calculating model activations'
        self.parser = parser
        self.model_loader = model_loader
        self.logger_loader = logger_loader

    def argparse(self, argparse):
        argparse.add_argument('model', help='model file')
        argparse.add_argument('files', metavar='FILE', nargs='+', help='input files')
        argparse.add_argument('-o/--output', dest='out', default='activations.h5', help='output hdf5 file (default: activations.h5)')
        argparse.add_argument('--batch-size', dest='batch_size', type=int, default=256, help='number of sequenes per batch (default: 256)')
        self.model_loader.argparse(argparse)
        self.parser.argparse(argparse)
        self.logger_loader.argparse(argparse)

    def __call__(self, args):
        import itertools
        with self.logger_loader(args) as logger:
            data, n_in, n_out = self.parser(args.files, args, ids=True)
            ids,data = zip(*data)
            model, model_name, start_epoch = self.model_loader(args.model, args, n_in, n_out)
            with pd.get_store(args.out, mode='w', complib='blosc', complevel=9) as store:
                acts = model.activations(data, batch_size=args.batch_size, callback=logger.progress_monitor())
                first = True
                for ident,act in itertools.izip(ids, acts):
                    if first:
                        template = ''.join(['{}']*(len(ident.colnames())+1))
                        print >>logger, template.format(*(ident.colnames()+['Layer']))
                        logger.flush()
                        first = False
                    for i in xrange(len(act)):
                        print >>logger, template.format(*(ident.cols()+[i]))
                        logger.flush()
                        layer = act[i][::ident.step]
                        for j in xrange(layer.shape[1]):
                            df = pd.DataFrame({'Position': range(ident.start, ident.start+layer.shape[0])
                                              , 'Activation': layer[:,j]})
                            for col,name in zip(ident.cols(), ident.colnames()):
                                df[name] = col
                            df['Layer'] = i
                            df['Unit'] = j
                            df.set_index(ident.colnames()+['Position','Layer','Unit'], inplace=True)
                            store.append('df', df, min_itemsize=30)
                """
                    layers = {'Layer{}'.format(j) : pd.DataFrame(act[j][::ident.step]) for j in xrange(len(act))}
                    df = pd.concat(layers, axis=1, names=['Layer', 'Unit'])
                    for col,name in zip(ident.cols(), ident.colnames()):
                        df[name] = col
                    df['Position'] = range(ident.start, ident.start+len(df))
                    df.set_index(ident.colnames()+['Position'], inplace=True)
                    store.append('df', df, min_itemsize=30) #use 30 -- should be chosen more intelligently
                """

        """
            first = True
            i = 0
            for acts in model.activations(data, batch_size=args.batch_size, callback=logger.progress_monitor()):
                ident = ids[i]
                if first:
                    header = ident.colnames()
                    n = sum(a.shape[1] for a in acts) + len(header) + 1
                    template = '\t'.join(['{}']*n)
                    layers = ['Layer{}-{}'.format(k,j) for k in xrange(len(acts)) for j in xrange(acts[k].shape[1])]
                    print >>logger, template.format(*(header+['Position']+layers))
                    first = False
                j = ident.start
                acts = [act[::ident.step] for act in acts]
                for jacts in itertools.izip(*acts):
                    layers = [jacts[l][k] for l in xrange(len(jacts)) for k in xrange(jacts[l].size)]
                    line = ident.cols() + [j] + layers
                    print >>logger, template.format(*line)
                    j += 1
                i += 1
        """
                


