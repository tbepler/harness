import os
import imp
import cPickle as pickle

def load_model(path, inputs):
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.py': #need to load model from python file
        model = imp.load_source("model", path)
        return model.build(inputs)
    else: #unpickle model from binary file
        with open(path) as f:
            return pickle.load(f)
