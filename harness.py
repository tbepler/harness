import argparse
import sys
import inspect

import command.train as train
import command.evaluate as evaluate
import command.annotate as annotate
import command.bin as bin

def main():
    parser = argparse.ArgumentParser("Harness for training and evaluating models.")
    subparsers = parser.add_subparsers()
    commands = [train, evaluate, annotate, bin]
    for cmd in commands:
        cmd_parser = subparsers.add_parser(cmd.name, description=cmd.description)
        cmd.init_parser(cmd_parser)
        cmd_parser.set_defaults(func=cmd.run)
    args = parser.parse_args()
    args.func(args)

if __name__=='__main__':
    main()
