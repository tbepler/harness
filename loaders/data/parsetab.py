
# parsetab.py
# This file is automatically generated. Do not edit.
_tabversion = '3.8'

_lr_method = 'LALR'

_lr_signature = '84C908854ED850503055FB89680E4D8C'
    
_lr_action_items = {'INT':([2,],[12,]),'FASTA_HEADER':([0,1,5,6,9,11,14,],[5,-6,-7,-4,5,-8,-5,]),'DNA':([0,1,3,5,7,11,12,13,],[2,11,-10,-7,2,-8,-12,-11,]),'$end':([1,3,4,5,6,7,8,9,10,11,12,13,14,],[-6,-10,-1,-7,-4,-9,0,-3,-2,-8,-12,-11,-5,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'fasta_seq_builder':([0,9,],[1,1,]),'label_seq':([0,7,],[3,13,]),'fasta_file':([0,],[4,]),'fasta_seq':([0,9,],[6,14,]),'labeled_file_builder':([0,],[7,]),'data':([0,],[8,]),'fasta_file_builder':([0,],[9,]),'labeled_file':([0,],[10,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> data","S'",1,None,None,None),
  ('data -> fasta_file','data',1,'p_data','parser.py',70),
  ('data -> labeled_file','data',1,'p_data','parser.py',71),
  ('fasta_file -> fasta_file_builder','fasta_file',1,'p_fasta_file','parser.py',75),
  ('fasta_file_builder -> fasta_seq','fasta_file_builder',1,'p_fasta_file_builder_start','parser.py',87),
  ('fasta_file_builder -> fasta_file_builder fasta_seq','fasta_file_builder',2,'p_fasta_file_builder_extend','parser.py',91),
  ('fasta_seq -> fasta_seq_builder','fasta_seq',1,'p_fasta_seq','parser.py',97),
  ('fasta_seq_builder -> FASTA_HEADER','fasta_seq_builder',1,'p_fasta_seq_builder','parser.py',103),
  ('fasta_seq_builder -> fasta_seq_builder DNA','fasta_seq_builder',2,'p_fasta_seq_builder','parser.py',104),
  ('labeled_file -> labeled_file_builder','labeled_file',1,'p_labeled_file','parser.py',113),
  ('labeled_file_builder -> label_seq','labeled_file_builder',1,'p_labeled_file_builder_start','parser.py',126),
  ('labeled_file_builder -> labeled_file_builder label_seq','labeled_file_builder',2,'p_labeled_file_builder_extend','parser.py',130),
  ('label_seq -> DNA INT','label_seq',2,'p_labeled_seq','parser.py',136),
]
