
import logging
import os
import sys
import codecs
import argparse

from lightner.commands.decoder import decode

def main():
    """
    Main function for terminal enterpoint.
    """

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            "decode": decode()
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)

    args = parser.parse_args()

    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()
