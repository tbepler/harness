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
            first = True
            i = 0
            for acts in model.activations(data, batch_size=args.batch_size, callback=logger.progress_monitor()):
                ident = ids[i]
                if first:
                    n = sum(a.shape[1] for a in acts) + 2
                    template = '\t'.join(['{}']*n)
                    layers = ['Layer{}-{}'.format(k,j) for k in xrange(len(acts)) for j in xrange(acts[k].shape[1])]
                    print >>logger, template.format('Sequence', 'Position', *layers)
                    first = False
                j = 1
                for jacts in itertools.izip(*acts):
                    layers = [jacts[l][k] for l in xrange(len(jacts)) for k in xrange(jacts[l].size)]
                    print >>logger, template.format(ident, j, *layers)
                    j += 1
                i += 1

                


