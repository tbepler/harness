import random

import harness.model as m
import harness.savers as s
import harness.logger as lg

class Fit(object):
    def __init__(self, parser, model_loader=m.Loader(), saver_loader=s.Loader()
                 , logger_loader=lg.Loader()):
        self.name = 'fit'
        self.description = 'Command for fitting models to data'
        self.parser = parser
        self.model_loader = model_loader
        self.saver_loader = saver_loader
        self.logger_loader = logger_loader

    def argparse(self, argparse):
        argparse.add_argument('model', help='model file')
        argparse.add_argument('files', metavar='FILE', nargs='+', help='input files')
        argparse.add_argument('--validate', dest='validate', type=float, default=0.05, help='fraction of data to keep for validation (default: 0.05)')
        argparse.add_argument('--max-iters', dest='max_iters', type=int, default=100, help='maximum number of iterations through the training data (default: 100)')
        argparse.add_argument('--batch-size', dest='batch_size', type=int, default=256, help='number of sequenes per batch (default: 256)')
        self.model_loader.argparse(argparse)
        self.parser.argparse(argparse)
        self.saver_loader.argparse(argparse)
        self.logger_loader.argparse(argparse)

    def __call__(self, args):
        with self.logger_loader(args) as logger:
            data, n_in, n_out = self.parser(args.files, args, ids=False)
            model, model_name, start_epoch = self.model_loader(args.model, args, n_in, n_out)
            if args.validate > 0:
                assert args.validate < 1
                random.shuffle(data)
                n = int(len(data)*args.validate)
                data_val = data[:n]
                data_train = data[n:]
            else:
                data_val = None
                data_train = data
            print >>logger, 'Training samples:', len(data_train)
            if data_val is not None:
                print >>logger, 'Validation samples:', len(data_val)
            saver = self.saver_loader(args, model_name)
            max_iters = args.max_iters
            epoch = start_epoch
            for info in model.fit(data_train, validate=data_val, batch_size=args.batch_size
                                  , max_iters=args.max_iters
                                  , callback=logger.progress_monitor()):
                epoch += 1
                saver.save(model, epoch, info['Loss'])
                template = '\t'.join(['{}']*(len(info)+1))
                if epoch == start_epoch + 1:
                    header = template.format('Epoch', *info.keys())
                    print >>logger, header
                line = template.format(epoch, *info.values())
                print >>logger, line
                

if __name__=='__main__':
    import harness.loaders.data.loader as d
    main = Fit(d.Parser())
    import argparse
    parser = argparse.ArgumentParser(description=main.description)
    main.argparse(parser)
    args = parser.parse_args()
    main(args)


