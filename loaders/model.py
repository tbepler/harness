import os
import imp
import math
import cPickle as pickle
import numpy as np

def init_parser(parser):
    parser.add_argument('--model-inputs', dest='model_inputs', type=int, default=0, help='number of inputs to the model, 0 or less mean infer from data (default: 0)')
    parser.add_argument('--model-outputs', dest='model_outputs', type=int, default=0, help='number of outputs from the model, 0 or less mean infer from data (default: 0)')

def load(path, args, n_in, n_out):
    ext = os.path.splitext(path)[-1].lower()
    if args.model_inputs > 0:
        n_in = args.model_inputs
    if args.model_outputs > 0:
        n_out = args.model_outputs    
    if ext == '.py': #need to load model from python file
        m = load_from_source(path, n_in, n_out)
    else: #unpickle model from binary file
        with open(path) as f:
            m = pickle.load(f)
    name,epoch = model_name_epoch(path)
    return m, name, epoch

def load_from_source(path, n_in, n_out):
    model = imp.load_source("model", path)
    return model.model(n_in, n_out)

def model_name_epoch(path):
    base = os.path.splitext(os.path.basename(path))[0]
    splt = base.split('epoch')
    if len(splt) > 1:
        name = splt[0][:-1]
        epoch = int(splt[1].split('_')[0])
    else:
        name = splt[0]
        epoch = 0
    return name,epoch

