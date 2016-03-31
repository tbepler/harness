import numpy as np

dna_encoder = {'A':0, 'C':1, 'G':2, 'T':3}

def read_fasta(f, dtype=np.int32, encoder=dna_encoder):
    name = None
    seq = []
    for line in f:
        line = line.strip()
        if not line == '':
            if line.startswith('>'):
                if name is not None:
                    yield name, np.array(seq, dtype=dtype)
                name = line[1:]
                seq = []
            else:
                seq.extend([encoder[c] for c in line])
    if name is not None:
        yield name, np.array(seq, dtype=dtype)
    
