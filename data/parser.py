import numpy as np
import ply.lex as lex

class Data(object):
    def __init__(self, x, y=None, name=None):
        self.x = x
        self.y = y
        self.name = name

    def __len__(self):
        return len(self.x)

class Lexer(object):
    tokens = (
        'DNA'
        , 'INT'
        , 'FASTA_HEADER'
    )

    def t_DNA(self, t):
        r'[ACGT]+'
        encoder = {'A':0, 'C':1, 'G':2, 'T':3}
        t.value = np.array([encoder[x] for x in t.value], dtype=np.int32)
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
                | label_file'''
        p[0] = p[1]

    def p_fasta_file_start(self, p):
        'fasta_file : fasta_seq'
        (seq,name) = p[1]
        p[0] = Data([seq], name=[name])

    def p_fasta_file_extend(self, p):
        'fasta_file : fasta_file fasta_seq'
        seq,name = p[2]
        p[1].x.append(seq)
        p[1].name.append(name)
        p[0] = p[1]

    def p_fasta_seq(self, p):
        'fasta_seq : fasta_seq_builder'
        name, seq = p[1]
        seq = np.concatenate(seq)
        p[0] = (seq, name)
    
    def p_fasta_seq_builder(self, p):
        '''fasta_seq_builder : FASTA_HEADER 
                             | fasta_seq_builder DNA'''
        if len(p) > 2:
            name, seq = p[1]
            seq.append(p[2])
            p[0] = (name, seq)
        else:
            p[0] = p[1], []

    def p_labeled_file(self, p):
        '''label_file : label_seq
                      | label_file label_seq'''
        if len(p) > 2:
            x,y = p[2]
            p[1].x.append(x)
            p[1].y.append(y)
            p[0] = p[1]
        else:
            x,y = p[1]
            p[0] = Data([x], y=[y])

    def p_labeled_seq(self, p):
        'label_seq : DNA INT'
        p[0] = (p[1], p[2])
        
    def p_error(self, t):
        print "Syntax error at {}".format(t)

    def __init__(self, lexer=Lexer(), **kwargs):
        self.lexer = lexer
        self.tokens = lexer.tokens
        self.parser = yacc.yacc(module=self, **kwargs)

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
    data = parser.parse(fasta_valid)
    for i in xrange(len(data)):
        print '>{}'.format(data.name[i])
        seq = data.x[i]
        print seq
        print seq.dtype

