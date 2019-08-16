"""Personal Formatting on Loguru"""

__version__ = "0.0.3"
__author__ = "Aditya Kelvianto Sidharta"

import logging
import os
import sys

from loguru import logger as loguru_logger

from logsensei.utils import _get_datetime


class Logger:
    """
    Setting up logger for the project. The log will be logged within the file as well
    logger.setup_logger(script_name) must be called first before using the logger.
    """

    def __init__(self):
        self.name = None
        self.datetime = None
        self.level = None
        self.sys_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> |" \
                          " <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " \
                          "<level>{message}</level>"
        self.file_format = None
        self.logger = loguru_logger
        self.logger.remove(0)
        self.logger.add(sys.stderr, format=self.sys_format, level=self.level)
        self.logger.patch(lambda record: record.update(name="my_module"))
        self.file_index = None
        self.template = {}

    def setup(self, name, logger_file, level=logging.DEBUG):
        self.name = name
        self.datetime = _get_datetime()
        self.level = level

        if self.file_index is not None:
            self.logger.remove(self.file_index)
        self.file_format = "<green>{time:YYYY-MM-DD HH:mm:ss:SSS}</green> | <level>{level: <8}</level> |" \
                           " <cyan>{name: ^15}</cyan>:<cyan>{function: ^15}</cyan>:<cyan>{line: >3}</cyan> - " \
                           "<level>{message}</level>"
        self.file_index = self.logger.add(
            os.path.join(logger_file, "{}_{}.log".format(self.name, self.datetime)),
            format=self.file_format,
            level=self.level,
        )

    def create_template(self, template_name, msg):
        if template_name in self.template.keys():
            self.logger.warning("Replacing the template message in {}".format(template_name))
        self.template[template_name] = msg

    def debug(self, msg):
        return self.logger.opt(depth=1).debug(msg)

    def info(self, msg):
        return self.logger.opt(depth=1).info(msg)

    def error(self, msg):
        return self.logger.opt(depth=1).error(msg)

    def warning(self, msg):
        return self.logger.opt(depth=1).warning(msg)

    def df(self, df):
        raise NotImplementedError

    def array(self, array):
        raise NotImplementedError

    def tensor(self, tensor):
        raise NotImplementedError

    def dict(self, dictionary):
        raise NotImplementedError

    def list(self, input_list):
        raise NotImplementedError

    def set(self, input_set):
        raise NotImplementedError

    def savepath(self, save_path):
        raise NotImplementedError

    def loadpath(self, load_path):
        raise NotImplementedError

    def scikit(self, model):
        raise NotImplementedError

    def xgboost(self, model):
        raise NotImplementedError

    def pytorch_tensor(self, model):
        raise NotImplementedError

    def tensorflow_tensor(self, model):
        raise NotImplementedError

    def pytorch_model(self, model):
        raise NotImplementedError

    def tensorflow_model(self, model):
        raise NotImplementedError

    def regression(self, true_array, predict_array):
        raise NotImplementedError

    def classification(self, true_array, predict_array):
        raise NotImplementedError

    def multiclass(self, true_array, predict_array):
        raise NotImplementedError

logger = Logger()
