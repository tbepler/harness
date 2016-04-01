import harness.model as m
import harness.logger as lg

class Loss(object):
    def __init__(self, parser, model_loader=m.Loader(), logger_loader=lg.Loader()):
        self.name = 'loss'
        self.description = 'Command for calculating the model loss on data'
        self.parser = parser
        self.model_loader = model_loader
        self.logger_loader = logger_loader

    def argparse(self, argparse):
        argparse.add_argument('model', help='model file')
        argparse.add_argument('files', metavar='FILE', nargs='+', help='input files')
        argparse.add_argument('--batch-size', dest='batch_size', type=int, default=256, help='number of sequenes per batch (default: 256)')
        argparse.add_argument('--no-aggregation', dest='no_aggregation', help='flag that indicates to report loss per data point rather than for the whole data set', action='store_true')
        self.model_loader.argparse(argparse)
        self.parser.argparse(argparse)
        self.logger_loader.argparse(argparse)

    def __call__(self, args):
        with self.logger_loader(args) as logger:
            if args.no_aggregation:
                data, n_in, n_out = self.parser(args.files, args, ids=True)
                model, model_name, start_epoch = self.model_loader(args.model, args, n_in, n_out)
                first = True
                for x in data:
                    ident = x[0]
                    info = model.loss(x[1:], batch_size=args.batch_size
                                      , callback=logger.progress_monitor())
                    if first:
                        template = '\t'.join(['{}']*(len(info)+1))
                        print >>logger, template.format('Id', *info.keys())
                        first = False
                    print >>logger, template.format(ident, *info.values())
            else:
                data, n_in, n_out = self.parser(args.files, args, ids=False)
                model, model_name, start_epoch = self.model_loader(args.model, args, n_in, n_out)
                info = model.loss(data, batch_size=args.batch_size
                                  , callback=logger.progress_monitor())
                template = '\t'.join(['{}']*len(info))
                print >>logger, template.format(*info.keys())
                print >>logger, template.format(*info.values())

                


