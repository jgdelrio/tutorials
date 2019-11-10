import os
import sys
import time
import errno
import inspect
import logging
import logging.handlers
import datetime as dt
from functools import wraps
from os.path import *


def logger(log_lvl="INFO"):
    log_obj = logging.getLogger("main")
    levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    log_obj.setLevel(levels[log_lvl])

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(levels[log_lvl])
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    log_obj.addHandler(ch)


__INSTANCE = None     # lazy-assigned instance
loggers = {}


class MyFormatter(logging.Formatter):
    converter = dt.datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s


class Logger:
    def __init__(self, module_name, filename, file_path=None, level=None, verbose=False):
        global loggers

        # Initial module name and level
        if module_name is None:
            module_name = 'module_name'

        if level is None:
            level = os.environ.get('COLOREDLOGS_LOG_LEVEL') or 'INFO'  # need to get the level from log_config_prop

        self.verbose = verbose

        if loggers.get(module_name):
            self.logger = loggers.get(module_name)
        else:
            logging
            self.logger = logging.getLogger(module_name)
            self.logger._warn_preinit_stderr = 0
            self.logger.setLevel(eval("logging." + level))
            # create file handler which logs even debug messages
            self.file_logger = self.save_file(filename, file_path)
            loggers[module_name] = self.logger

    def save_file(self, file_name=None, file_path=None):
        file_logger = FileLogger(self.logger, file_name, file_path)
        return file_logger

    def debug(self, msg):
        self.logger.debug(msg)
        self.print_verbose(msg)

    def info(self, msg):
        self.logger.info(msg)
        self.print_verbose(msg)

    def warn(self, msg):
        self.logger.warning(msg)
        self.print_verbose(msg)

    def error(self, msg):
        self.logger.error(msg)
        self.print_verbose(msg)

    def critical(self, msg):
        self.logger.critical(msg)
        self.print_verbose(msg)

    def print_verbose(self, msg):
        if self.verbose:
            print(msg)


class FileLogger:
    def __init__(self, logger, file_name, file_path=None):
        if file_path is None:
            file_path = scriptinfo()["dir"]

        self.create_log_dir(file_path)
        # Check file name
        file_name = check_filename_extension(file_name)

        # Initial
        f_path = os.path.join(file_path, file_name)
        # self.handler = logging.FileHandler(file_path)
        self.handler = logging.handlers.RotatingFileHandler(f_path, maxBytes=2097152, backupCount=3)

        # create a formatter and set the formatter for the handler
        self.formatter = MyFormatter(fmt='%(asctime)s,%(name)s,%(message)s',
                                     datefmt='%Y-%m-%d,%H:%M:%S')
        # self.formatter = logging.Formatter('%(asctime)s,%(name)s,%(message)s')
        self.handler.setFormatter(self.formatter)
        logger.addHandler(self.handler)
        self.logger = logger

    def open_stream(self):
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(self.logger.level)
        stream.setFormatter(self.handler)
        self.logger.addHandler(stream)

    def __del__(self):
        # Close the log file
        # self.handler.close()
        x = logging._handlers.copy()
        for i in x:
            self.logger.removeHandler(i)
            i.flush()
            i.close()

    @staticmethod
    def create_log_dir(file_path):
        # Check logs directory and create file

        try:
            os.makedirs(file_path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(file_path):
                pass
            else:
                raise


def error_message(logger, msg):
    msg = '\n===================ERROR===================\n' + msg
    msg += '\n==========================================='
    logger.error(msg)


def check_filename_extension(file_name):
    file_level = file_name.split('.')
    if file_level[len(file_level) - 1] not in 'log':
        file_name += '.log'
    return file_name


def logger_instance(logclass='ml', filename='default', level='INFO', defer=False):
    """
    Creates an instance of the logger class
    :param logclass: name to be save as reference in the logger
    :param filename: filename of the output log
    :param defer:    defers the creation of the instance
    :return:         logger instance
    """
    global __INSTANCE

    if defer:
        return

    if __INSTANCE is None:
        if logclass is None:
            raise Exception("Missing 'logclass' parameter")

        if filename is None:
            raise Exception("Missing 'filename' parameter")

        __INSTANCE = Logger(logclass, filename, level)
        return __INSTANCE

    else:
        return __INSTANCE


def logger_wrapper(orig_func, file_path=None):
    """
    :param orig_func: function in which the wrapper will be enabled
    :return:          None
    Use it before the definition of the function:
        @logger_wrapper
        @timer_wrapper
        def my_function()
    """
    if file_path is None:
        file_path = scriptinfo()["dir"]
    module = inspect.getmodule(orig_func)
    module_name = splitext(module.__name__)[1][1:]
    class_name = splitext(orig_func.__qualname__)[0]

    logging.basicConfig(filename=os.path.join(file_path, '{}.log'.format(module_name + '.' + class_name)),
                        level=logging.INFO)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(
            'Run with args: {}, and kwargs: {}'.format(args, kwargs))
        return orig_func(*args, **kwargs)

    return wrapper


def timer_wrapper(orig_func):
    """
    :param orig_func: function in which the wrapper will be enabled
    :return:          None
    Use it before the definition of the function:
        @logger_wrapper
        @timer_wrapper
        def my_function()
    """

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in: {} sec'.format(orig_func.__name__, t2))
        return result

    return wrapper


def scriptinfo():
    '''
    Returns a dictionary with information about the running top level Python
    script:
    ---------------------------------------------------------------------------
    dir:    directory containing script or compiled executable
    name:   name of script or executable
    source: name of source code file
    ---------------------------------------------------------------------------
    "name" and "source" are identical if and only if running interpreted code.
    When running code compiled by py2exe or cx_freeze, "source" contains
    the name of the originating Python script.
    If compiled by PyInstaller, "source" contains no meaningful information.
    '''

    import os, sys, inspect
    # ---------------------------------------------------------------------------
    # scan through call stack for caller information
    # ---------------------------------------------------------------------------
    for teil in inspect.stack():
        # skip system calls
        if teil[1].startswith("<"):
            continue
        if teil[1].upper().startswith(sys.exec_prefix.upper()):
            continue
        trc = teil[1]

    # trc contains highest level calling script name
    # check if we have been compiled
    if getattr(sys, 'frozen', False):
        scriptdir, scriptname = os.path.split(sys.executable)
        return {"dir": scriptdir,
                "name": scriptname,
                "source": trc}

    # from here on, we are in the interpreted case
    scriptdir, trc = os.path.split(trc)
    # if trc did not contain directory information,
    # the current working directory is what we need
    if not scriptdir:
        scriptdir = os.getcwd()

    scr_dict = {"name": trc,
                "source": trc,
                "dir": scriptdir}
    return scr_dict
