
import logging
import sys

from . import project

# Resources:
# https://stackoverflow.com/questions/39492471/how-to-extend-the-logging-logger-class
# https://www.toptal.com/python/in-depth-python-logging
# https://zetcode.com/python/logging/
# https://coralogix.com/blog/python-logging-best-practices-tips/


loggers = {}

class SubsystemNameFormatter(logging.Formatter):
    """
    Strip the top-level package prefix: 
        e.g. 'clorinn.optimize.frankwolfe' -> 'optimize.frankwolfe'.
    """
    def format(self, record):
        parts = record.name.split('.', 1)
        record.name = parts[1] if len(parts) > 1 else parts[0]
        return super().format(record)


class ShortNameFormatter(logging.Formatter):
    """
    Keep only the module filename:
        e.g. 'clorinn.optimize.frankwolfe' -> 'frankwolfe'.
    """
    def format(self, record):
        record.name = record.name.split('.')[-1]
        return super().format(record)


def get_new_formatter(fmt = project.logging_format()):
    #return logging.Formatter(fmt)
    return ShortNameFormatter(fmt)


def get_new_handler(formatter, logfile = None):
    if logfile is None: # log to stdout
        handler = logging.StreamHandler(sys.stdout)
    else: # log to file
        handler = logging.FileHandler(logfile)
    handler.setFormatter(formatter)
    return handler

def get_loglevel(verbose=0):
    if verbose >= 2:
        return logging.DEBUG
    if verbose == 1:
        return logging.INFO
    # do not inherit root logger level if verbose = 0
    # explicitly silence the logs when the user wants.
    return logging.WARN   


class CustomLogger(logging.getLoggerClass()):

    global loggers

    def __init__(self, name, fmt = None, level = None, logfile = None, is_handler = False, is_debug = False):
        # Make sure the project has a root logger
        # after introducing _setup_root_logger(),
        # this is now dead code as the first condition is always False.
        # retained as a safety fallback in case loggers is ever cleared externally.
        if project.get_name() not in loggers.keys() and not name == project.get_name():
            self.create_default_logger()
        # Create the logger
        self.create(name, fmt = fmt, level = level, logfile = logfile, is_handler = is_handler, is_debug = is_debug)
        return

    def __repr__(self):
        if loggers.keys():
            m = max(map(len, list(loggers.keys()))) + 1
            return '\n'.join([k.rjust(m) + ':' + repr(v)
                              for k, v in loggers.items()])
        else:
            return self.__class__.__name__ + "()"


    def __dir__(self):
        return list(loggers.keys())


    def create_default_logger(self):
        self.create(project.get_name(), level = project.logging_level(),
            fmt = project.logging_format(), logfile = project.logging_file(),
            is_handler = True)
        return


    def override_global_default_loglevel(self, level):
        base_logger = loggers[project.get_name()]
        if level is None: level = base_logger.getEffectiveLevel()
        if not base_logger.getEffectiveLevel() == level:
            base_logger.setLevel(level)
        return


    def override_subsystem_loglevel(self, level):
        """
        stop using the root clorinn logger as a global dial 
        and instead scope each subsystem to its own sub-hierarchy:

            clorinn           ← root, set once at startup, never touched again
            clorinn.optimize  ← all solvers propagate here
            clorinn.tests     ← all tests propagate here
            clorinn.utils     ← utils propagate here
            ...

        Edge case
        ---------
        If a top-level logger named exactly 'clorinn' itself calls this function,
        then `split('.')[1]' would raise IndexError - but that logger is created by 
        `_setup_root_logger` and never calls `override_subsystem_loglevel`, 
        so it cannot arise in practice.
        """
        subsystem = self.logger.name.split('.')[1]
        name = f"{project.get_name()}.{subsystem}"
        subsystem_logger = self.register_logger(name) 
        if level is not None:
            subsystem_logger.setLevel(level)


    def set_loglevel(self, level):
        if level is None: level = self.logger.parent.getEffectiveLevel()
        if not self.logger.getEffectiveLevel() == level:
            self.logger.setLevel(level)
        return


    def create(self, name, fmt = None, level = None, logfile = None, is_handler = False, is_debug = False):
        # Register new logger if not already present.
        # A logger is unique by name, meaning that if a logger with the name `foo` 
        # has been created, the consequent calls of `logging.getLogger("foo")` 
        # will return the same object.
        is_new = True if not loggers.get(name) else False

        self.logger = self.register_logger(name)

        # The log level can be altered during runtime.
        # Inherit parent logging level if level = None
        level = level if not is_debug else logging.DEBUG
        self.set_loglevel(level)

        # Python loggers form a hierarchy. A logger named main is a parent of main.new.
        # Child loggers propagate messages up to the handlers associated with their 
        # ancestor loggers. Because of this, it is unnecessary to define and configure 
        # handlers for all the loggers in the application. It is sufficient to 
        # configure handlers for a top-level logger and create child loggers as needed.
        # 
        # If a child logger gets a new handler, then the same information will be 
        # processed by the handlers of the child logger and the parent logger.
        if is_handler:
            formatter = get_new_formatter(fmt)
            handler   = get_new_handler(formatter, logfile)
            self.logger.addHandler(handler)

        if is_new:
            self.logger.debug(f"Created {self.logger}, Parent: {self.logger.parent}")

        return


    def info(self, msg, extra=None):
        self.logger.info(msg, extra=extra)


    def error(self, msg, extra=None):
        self.logger.error(msg, extra=extra)


    def debug(self, msg, extra=None):
        self.logger.debug(msg, extra=extra)


    def warn(self, msg, extra=None):
        self.logger.warning(msg, extra=extra)


    def register_logger(self, name):
        newlogger = loggers.get(name)
        if not newlogger:
            newlogger = logging.getLogger(name)
            loggers[name] = newlogger
        return newlogger


def _setup_root_logger():
    """
    Initialise the root logger at import time so that module-level
    loggers (logging.getLogger(__name__)) have a handler available
    regardless of whether any class has been instantiated yet.
    """
    name = project.get_name()
    if name not in loggers:
        formatter = get_new_formatter(project.logging_format())
        handler   = get_new_handler(formatter, project.logging_file())
        logger    = logging.getLogger(name)
        logger.setLevel(project.logging_level())
        logger.addHandler(handler)
        loggers[name] = logger

_setup_root_logger()
