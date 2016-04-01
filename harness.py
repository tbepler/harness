
class Harness(object):
    def __init__(self, cmds=[]):
        self.cmds = cmds

    def argparse(self, parser):
        subparsers = parser.add_subparsers()
        for cmd in self.cmds:
            cmd_parser = subparsers.add_parser(cmd.name, description=cmd.description)
            cmd.argparse(cmd_parser)
            cmd_parser.set_defaults(func=cmd)

    def __call__(self, args):
        args.func(args)

