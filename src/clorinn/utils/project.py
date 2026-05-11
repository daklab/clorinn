import os
import logging

def name():
    return "Clorinn"


def description():
    return "Low rank matrix approximation using nuclear norm constraint"


def version():
    ## Get the version from version.py without importing the package
    vfn = os.path.join(os.path.dirname(__file__), '../version.py')
    namespace = {}
    with open(vfn, "r", encoding="utf-8") as f:
        exec(compile(f.read(), vfn, "exec"), namespace)
    return namespace["__version__"]


def get_name():
    namestr = __name__
    return namestr.strip().split('.')[0].strip()


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0
