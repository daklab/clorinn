"""
An aggressively minimal CLI support. 

It does not try to expose the fitting API, it is only intended
to be used for a few lightweight things.

    clorinn --version
    clorinn --test
    
These are harmless and useful for sanity checks after installation.
A working `clorinn --version` is useful because it immediately tells
that the installed package is importable and the entry point is wired 
correctly.

"""
import sys 
import argparse

from .utils.logs import configure_logging
from .utils import project


def parse_args():
    parser = argparse.ArgumentParser(description=project.description())
    parser.add_argument('--test',
                        dest = 'test',
                        action = 'store_true',
                        help = 'Perform unit tests')
    parser.add_argument('--testmodule',
                        dest = 'testmodules',
                        metavar = 'MODULES',
                        nargs = "*",
                        type = str,
                        default = None,
                        help = 'Names of modules to be tested')
    parser.add_argument('--version',
                        dest = 'version',
                        action = 'store_true',
                        help = 'Print version number')
    parser.add_argument('--verbose',
                        dest = 'verbose',
                        action = 'store_true',
                        help = 'Print information while running tests')
    parser.add_argument('--vverbose',
                        dest = 'vverbose',
                        action = 'store_true',
                        help = 'Print more information while running tests')
    res = parser.parse_args()
    return parser, res 


def show_version():
    print ("")
    print ("{:s} version {:s}".format(project.name(), project.version()))
    print ("")
    return


def show_help(parser, opts):
    parser.print_help(sys.stderr)
    sys.exit(1)
    return


def main():
    parser, opts = parse_args()

    verbosity = 2 if opts.vverbose else 1 if opts.verbose else 0
    mlogger  = configure_logging(verbose=verbosity, formatter="short", force=True)

    if opts.test or opts.testmodules:
        mlogger.debug("Calling logger from main")
        # update package logger to verbose=0, 
        # handle test logger from unittests
        mlogger = configure_logging(verbose = 0, formatter = "short", force = True)
        mlogger.debug("Package is now quiet. If you are seeing this message, there is a logger error.")
        from .tests.run import run_unittests
        run_unittests(test_class_names=opts.testmodules, verbosity=verbosity)

    elif opts.version:
        show_version()

    else:
        show_help(parser, opts)

if __name__ == "__main__":
    main()
