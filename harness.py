import argparse
import sys
import inspect

import command.train as train
import command.evaluate as evaluate
import command.annotate as annotate

def main():
    parser = argparse.ArgumentParser("Harness for training and evaluating models.")
    commands = {train.name:train, evaluate.name:evaluate, annotate.name:annotate}
    names = list(commands.keys())
    parser.add_argument('cmd', choices=names, help='Command to run.')
    args = parser.parse_args()
    cmd = commands[args.cmd]
    cmd.run(cmd.parser.parse_args(sys.argv[1:]))

if __name__=='__main__':
    main()
