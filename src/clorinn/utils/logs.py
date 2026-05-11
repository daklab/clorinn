import copy
import logging
import sys


PACKAGE_LOGGER = "clorinn"
SOLVER_SUBSYSTEMS = ("clorinn.optimize", "clorinn.utils")
DEFAULT_LOGGER_FORMAT = "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s"


def verbose_to_level(verbose):
    if verbose is None: return None
    if verbose >= 2:    return logging.DEBUG
    if verbose == 1:    return logging.INFO
    return logging.WARNING


def subsystem_name(logger_name):
    parts = logger_name.split(".")
    if len(parts) >= 2 and parts[0] == PACKAGE_LOGGER:
        return f"{parts[0]}.{parts[1]}"
    return logger_name


def get_logger(name, *, verbose=None, level=None, scope="package"):
    """
    Set the logging level for a Clorinn logger.

    scope:
        "module"    -> clorinn.optimize.frankwolfe
        "solver"    -> SOLVER_SUBSYSTEMS
        "subsystem" -> clorinn.optimize
        "package"   -> clorinn
    """
    if level is None:
        level = verbose_to_level(verbose)

    if level is None:
        return logging.getLogger(name)

    if scope == "package":
        target_names = (PACKAGE_LOGGER)
    elif scope == "solver":
        target_names = SOLVER_SUBSYSTEMS
    elif scope == "subsystem":
        target_names = (subsystem_name(name))
    elif scope == "module":
        target_names = (name)
    else:
        raise ValueError(f"Unknown logging scope: {scope!r}")

    for s in target_names:
        logger = logging.getLogger(s)
        logger.setLevel(level)

    return logging.getLogger(name)


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


def configure_logging(
    *,
    verbose=None,
    level=None,
    stream=None,
    filename=None,
    subsystem_levels=None,
    fmt=DEFAULT_LOGGER_FORMAT,
    formatter="short",
    force=False,
    propagate=True,
):
    """
    Optional caller-facing logging setup.

    This is the only place, apart from the CLI, where Clorinn should create
    handlers.
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER)

    if force:
        for handler in list(package_logger.handlers):
            if isinstance(handler, logging.NullHandler):
                continue
            package_logger.removeHandler(handler)

    if level is None:
        level = verbose_to_level(verbose)
    if level is None:
        level = logging.WARNING

    if filename is not None:
        handler = logging.FileHandler(filename)
    else:
        handler = logging.StreamHandler(stream if stream is not None else sys.stderr)

    if formatter == "short":
        log_formatter = ShortNameFormatter(fmt)
    elif formatter == "subsystem":
        log_formatter = SubsystemNameFormatter(fmt)
    else:
        log_formatter = logging.Formatter(fmt)

    handler.setLevel(logging.NOTSET)
    handler.setFormatter(log_formatter)
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    package_logger.propagate = propagate

    if subsystem_levels:
        for subsystem, value in subsystem_levels.items():
            name = subsystem if subsystem.startswith(PACKAGE_LOGGER) else f"{PACKAGE_LOGGER}.{subsystem}"
            logging.getLogger(name).setLevel(value)

    return package_logger
