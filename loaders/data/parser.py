import numpy as np
import ply.lex as lex

dna_encoder = {'A':0, 'C':1, 'G':2, 'T':3}

class Id(object):
    def __init__(self, name, start=None, end=None):
        self.name = name
        self.start = start
        self.end = end
        
    def __str__(self):
        if self.start is None and self.end is None:
            return str(self.name)
        elif self.start is None:
            return ''.join([str(self.name), '[:', str(self.end), ']'])
        elif self.end is None:
            return ''.join([str(self.name), '[', str(self.start), ':]'])
        else:
            return ''.join([str(self.name), '[', str(self.start), ':', str(self.end), ']'])

class Lexer(object):
    tokens = (
        'DNA'
        , 'INT'
        , 'FASTA_HEADER'
    )

    def t_DNA(self, t):
        r'[ACGT]+'
        t.value = np.array([dna_encoder[x] for x in t.value], dtype=np.int32)
        return t

    def t_INT(self, t):
        r'\d+'
        t.value = int(t.value)
        return t
    
    def t_FASTA_HEADER(self, t):
        r'>.*'
        t.value = t.value[1:] #strip the >
        return t

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    t_ignore = ' \t'

    def t_error(self, t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    def __init__(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)

    def input(self, data):
        self.lexer.input(data)

    def token(self):
        return self.lexer.token()
        
    def __iter__(self):
        return self.lexer

import ply.yacc as yacc

class Parser(object):
    def p_data(self, p):
        '''data : fasta_file
                | labeled_file'''
        p[0] = p[1]

    def p_fasta_file(self, p):
        'fasta_file : fasta_file_builder'
        xs = p[1]
        if self.window > 0:
            lst = []
            for n,x in xs:
                for i in xrange(0, len(x)-self.window+1, self.stride):
                    data = (Id(n.name, i+1, i+self.window), x[i:i+self.window])
                    lst.append(data)
            xs = lst
        p[0] = xs, len(dna_encoder), len(dna_encoder)
        
    def p_fasta_file_builder_start(self, p):
        'fasta_file_builder : fasta_seq'
        p[0] = [p[1]]

    def p_fasta_file_builder_extend(self, p):
        'fasta_file_builder : fasta_file_builder fasta_seq'
        xs = p[1]
        xs.append(p[2])
        p[0] = xs

    def p_fasta_seq(self, p):
        'fasta_seq : fasta_seq_builder'
        name, seq = p[1]
        seq = np.concatenate(seq)
        p[0] = (name, seq)
    
    def p_fasta_seq_builder(self, p):
        '''fasta_seq_builder : FASTA_HEADER 
                             | fasta_seq_builder DNA'''
        if len(p) > 2:
            name, seq = p[1]
            seq.append(p[2])
            p[0] = (name, seq)
        else:
            p[0] = Id(p[1]), []

    def p_labeled_file(self, p):
        'labeled_file : labeled_file_builder'
        xs = p[1]
        n_out = max(y for _,x,y in xs)
        if self.window > 0:
            lst = []
            for n,x,y in xs:
                for i in xrange(0, len(x)-self.window+1, self.stride):
                    data = (Id(n.name, i+1, i+self.window), x[i:i+self.window], y)
                    lst.append(data)
            xs = lst
        p[0] = (xs, len(dna_encoder), n_out)

    def p_labeled_file_builder_start(self, p):
        'labeled_file_builder : label_seq'
        p[0] = [p[1]]

    def p_labeled_file_builder_extend(self, p):
        'labeled_file_builder : labeled_file_builder label_seq'
        xs = p[1]
        xs.append(p[2])
        p[0] = xs

    def p_labeled_seq(self, p):
        'label_seq : DNA INT'
        self.counter += 1
        x = p[1]
        y = p[2]
        name = ''.join(['s', str(self.counter)])
        data = (Id(name), x, y)
        p[0] = data
        
    def p_error(self, t):
        print "Syntax error at {}".format(t)

    def __init__(self, lexer=Lexer(), window=0, stride=1):
        self.window = window
        self.stride = stride
        self.counter = 0 #sequence counter
        self.lexer = lexer
        self.tokens = lexer.tokens
        self.parser = yacc.yacc(module=self)

    def parse(self, data, **kwargs):
        return self.parser.parse(data, lexer=self.lexer, **kwargs)

if __name__=='__main__':
    test = 'ACGTATATATATATAT 0\n>fasta name\nACTGTGTGTATATATACCGG'
    lexer = Lexer()
    lexer.input(test)
    for tok in lexer:
        print tok

    parser = Parser()
    print parser.parse(test)

    fasta_valid = '''>sequence 1
                     ACGTATATAGATGATATATATATAT

                     >sequence 2
                     GGGGGAGAGAG
                     TAGCTACGTATATTATAGCGACACACTAC
                     ACGTACGTA

                     >sequence 3
                     TCACATCTACTACTAGCGACGACTACT'''
    data, n_in, n_out = parser.parse(fasta_valid)
    print 'n_in = {}, n_out = {}'.format(n_in, n_out)
    for name, x in data:
        print '>{}'.format(name)
        print x
        print x.dtype

    parser = Parser(window=5, stride=2)
    data, n_in, n_out = parser.parse(fasta_valid)
    print 'n_in = {}, n_out = {}'.format(n_in, n_out)
    for name, x in data:
        print '>{}'.format(name)
        print x
        print x.dtype

