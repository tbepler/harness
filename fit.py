import random

import loaders.data.loader as d
import loaders.model as m
import savers as s

def main(args):
    data, n_in, n_out = load_data(args.files, args)
    model, model_name, epoch = load_model(args.model, args, n_in, n_out)
    if args.validate > 0:
        assert args.validate < 1
        random.shuffle(data)
        n = int(len(data)*args.validate)
        data_val = data[:n]
        data_train = data[n:]
        
    else:
        data_val = None
        data_train = data
    print 'Training samples:', len(data_train)
    if data_val is not None:
        print 'Validation samples:', len(data_val)
    #split off sequence identifiers
    print 'Splittin labels'
    data_train = zip(*(zip(*data_train)[1:]))
    data_val = zip(*(zip(*data_val)[1:]))
    print 'Done splittin labels'
    saver = load_saver(args, model_name)
    #kwargs = {k:args[k] for k in ['batch_size', 'max_iters', 'verbose'] if args[k] is not None}
    for loss in model.fit(data_train, validate=data_val, batch_size=args.batch_size
                          , max_iters=args.max_iters, verbose=args.verbose):
        epoch += 1
        saver.save(model, epoch, loss)

def init_parser(parser):
    parser.add_argument('model', help='model file')
    parser.add_argument('files', metavar='FILE', nargs='+', help='input files')
    parser.add_argument('--validate', dest='validate', type=float, default=0.05, help='fraction of data to keep for validation (default: 0.05)')
    parser.add_argument('--max-iters', dest='max_iters', type=int, default=100, help='maximum number of iterations through the training data (default: 100)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=256, help='number of sequenes per batch (default: 256)')
    parser.add_argument('--verbose/-v', dest='verbose', type=int, default=1, help='verbosity level (default: 1)')
    d.init_parser(parser)
    m.init_parser(parser)
    s.init_parser(parser)
    
def load_model(path, args, n_in, n_out):
    return m.load(path, args, n_in, n_out)

def load_data(paths, args):
    return d.load(paths, args)

def load_saver(args, model_name):
    return s.get_saver(args, model_name)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Wrapper program for training models.")
    init_parser(parser)
    args = parser.parse_args()
    main(args)


