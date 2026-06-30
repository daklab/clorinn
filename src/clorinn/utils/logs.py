import copy
import logging
import sys

PACKAGE_LOGGER = __name__.split(".", 1)[0]
DEFAULT_LOGGER_FORMAT = "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def verbosity_to_level(verbosity):
    if verbosity is None: return None
    if verbosity >= 2:    return logging.DEBUG
    if verbosity == 1:    return logging.INFO
    return logging.WARNING


def subsystem_of(name):
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == PACKAGE_LOGGER:
        return f"{parts[0]}.{parts[1]}"
    return name


def set_loglevel(name, *, verbosity=None, level=None, scope="package"):
    """
    Set the logging level for a Clorinn logger.

    scope:
        "module"    -> clorinn.optimize.frankwolfe
        "subsystem" -> clorinn.optimize
        "package"   -> clorinn
    """
    if level is None:
        level = verbosity_to_level(verbosity)

    if level is None:
        return logging.getLogger(name)

    scope_map = {
        "package":   PACKAGE_LOGGER,
        "subsystem": subsystem_of(name),
        "module":    name,
    }
    try:
        target = scope_map[scope]
    except KeyError:
        raise ValueError(f"Unknown logging scope: {scope!r}")

    logging.getLogger(target).setLevel(level)

    return


def get_log_formatter(formatter="short"):
    fmt_classes = {
        "short" :     ShortNameFormatter,
        "subsystem" : SubsystemNameFormatter,
        "testclass" : TestClassNameFormatter,
    }
    fmt_class = fmt_classes.get(formatter, logging.Formatter)
    return fmt_class


class ShortNameFormatter(logging.Formatter):
    """
    Keep only the module filename:
        e.g. 'clorinn.optimize.frankwolfe' -> 'frankwolfe'.
    """
    def format(self, record):
        r = copy.copy(record)
        r.name = r.name.split(".")[-1]
        return super().format(r)


class SubsystemNameFormatter(logging.Formatter):
    """
    Strip the top-level package prefix:
        e.g. 'clorinn.optimize.frankwolfe' -> 'optimize.frankwolfe'.
    """
    def format(self, record):
        r = copy.copy(record)
        parts = r.name.split(".", 1)
        r.name = parts[1] if len(parts) > 1 else parts[0]
        return super().format(r)


class TestClassNameFormatter(logging.Formatter):
    """
    Replace name with TestClass:
        e.g. 'clorinn.tests.unittest_tester' -> 'TestAFWNNMCorrFull'.
    """
    def format(self, record):
        r = copy.copy(record)
        if hasattr(r, "test_name"):
            r.name = r.test_name
        else:
            r.name = r.name.split(".")[-1]
        return super().format(r)


def apply_verbosity(name, verbosity=None):
    if verbosity is None:
        return
    configure_logging() # idempotent install
    level = verbosity_to_level(verbosity)
    logging.getLogger(name).setLevel(level)
    return


def configure_module_logger(name, verbosity=None):
    subsystem = subsystem_of(name)
    apply_verbosity(subsystem, verbosity=verbosity)
    return logging.getLogger(name)


def configure_logging(*, 
    verbosity=None, level=None,
    stream=None, filename=None,
    subsystem_levels=None, subsystem_verbosity=None,
    fmt=DEFAULT_LOGGER_FORMAT, datefmt=DEFAULT_DATE_FORMAT, formatter="short",
    force=False, propagate=True):
    """
    Optional caller-facing logging setup.
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER)
    already_ours   = any(getattr(h, "_clorinn", False) for h in package_logger.handlers)

    # verbosity = None, level = None keeps level at None
    if level is None:
        level = verbosity_to_level(verbosity)

    if already_ours and not force:
        # we already created a package_logger 
        # update level, don't add another handler
        if level is not None:
            package_logger.setLevel(level)
        return package_logger

    if force:
        for handler in list(package_logger.handlers):
            if isinstance(handler, logging.NullHandler):
                continue
            package_logger.removeHandler(handler)

    if filename is not None:
        handler = logging.FileHandler(filename)
    else:
        handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler._clorinn = True

    log_formatter = get_log_formatter(formatter=formatter)

    handler.setLevel(logging.NOTSET)
    handler.setFormatter(log_formatter(fmt=fmt, datefmt=datefmt))
    package_logger.addHandler(handler)

    if level is None: level = logging.WARNING
    package_logger.setLevel(level)
    package_logger.propagate = propagate

    if subsystem_levels:
        for subsystem, value in subsystem_levels.items():
            target = subsystem if subsystem.startswith(PACKAGE_LOGGER) else f"{PACKAGE_LOGGER}.{subsystem}"
            logging.getLogger(target).setLevel(value)

    if subsystem_verbosity:
        for subsystem, value in subsystem_verbosity.items():
            target = subsystem if subsystem.startswith(PACKAGE_LOGGER) else f"{PACKAGE_LOGGER}.{subsystem}"
            logging.getLogger(target).setLevel(verbosity_to_level(value))

    return
