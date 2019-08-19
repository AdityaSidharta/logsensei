"""Personal Formatting on Loguru"""

__version__ = "0.0.3"
__author__ = "Aditya Kelvianto Sidharta"

import logging
import os
import sys
from collections import Counter

import numpy as np
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
        self.time_sys_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
        self.level_sys_format = "<level>{level}</level>"
        self.function_sys_format = (
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        )
        self.message_sys_format = "<level>{message}</level>"
        self.sys_format = "{} | {} | {} | {}".format(
            self.time_sys_format,
            self.level_sys_format,
            self.function_sys_format,
            self.message_sys_format,
        )
        self.time_file_format = None
        self.level_file_format = None
        self.function_file_format = None
        self.message_file_format = None
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
        self.time_file_format = "<green>{time:YYYY-MM-DD HH:mm:ss:SSS}</green>"
        self.level_file_format = "<level>{level: <8}</level>"
        self.function_file_format = "<cyan>{name: ^15}</cyan>:<cyan>{function: ^15}</cyan>:<cyan>{line: >3}</cyan>"
        self.message_file_format = "<level>{message}</level>"
        self.file_format = "{} | {} | {} | {}".format(
            self.time_file_format,
            self.level_file_format,
            self.function_file_format,
            self.message_file_format,
        )
        self.file_index = self.logger.add(
            os.path.join(logger_file, "{}_{}.log".format(self.name, self.datetime)),
            format=self.file_format,
            level=self.level,
        )

    def create_template(self, template_name, msg):
        if template_name in self.template.keys():
            self.logger.warning(
                "Replacing the template message in {}".format(template_name)
            )
        self.template[template_name] = msg

    def debug(self, msg):
        self.logger.opt(depth=1).debug(msg)

    def info(self, msg):
        self.logger.opt(depth=1).info(msg)

    def error(self, msg):
        self.logger.opt(depth=1).error(msg)

    def warning(self, msg):
        self.logger.opt(depth=1).warning(msg)

    def df(self, df, df_name):
        shape = df.shape
        self.info("DataFrame {} shape : {}".format(df_name, shape))
        self.info("DataFrame {} info:".format(df_name))
        self.info(df.info())

    def array(self, array, array_name):
        shape = array.shape
        self.info("Array {} shape : {}".format(array_name, shape))
        if array.ndim == 1:
            n_values = len(array)
            unique_values = set(array)
            n_unique_values = len(set(array))
            n_missing_values = np.sum(np.isnan(array))
            self.info("Array {} unique values : {}".format(array_name, unique_values))
            self.info("Array {} cardinality : {}".format(array_name, n_unique_values))
            self.info(
                "Array {} missing values : {} ({:.2f}%)".format(
                    array_name, n_missing_values, n_missing_values / n_values * 100.
                )
            )
            if (array.dtype == float) or (array.dtype == int):
                mean_value = np.nanmean(array)
                std_value = np.nanstd(array)
                max_value = np.nanmax(array)
                min_value = np.nanmin(array)
                median_value = np.nanmedian(array)
                perc_25_value = np.nanpercentile(array, 25)
                perc_75_value = np.nanpercentile(array, 75)
                self.info(
                    "Array {} info : MEAN={} | STD={} | MIN={} | 25TH={} | MEDIAN={} | 75TH={} | MAX={}".format(
                        array_name,
                        mean_value,
                        std_value,
                        min_value,
                        perc_25_value,
                        median_value,
                        perc_75_value,
                        max_value,
                    )
                )
            else:
                most_common = Counter(array).most_common(5)
                n_most_common = len(most_common)
                self.info(
                    "Array {} top {} values : ".format(array_name, n_most_common)
                    + " | ".join(
                        [
                            "{} - {}({:.2f}%)".format(
                                x[0], x[1], x[1] / n_values * 100.
                            )
                            for x in most_common
                        ]
                    )
                )

    def compare(self, array_1, array_2, array_1_name, array_2_name):
        array_1_shape = array_1.shape
        array_2_shape = array_2.shape
        array_1_ndim = array_1.ndim
        array_2_ndim = array_2.ndim
        self.info(
            "Compare {} vs {} - shape : {} vs {}".format(
                array_1_name, array_2_name, array_1_shape, array_2_shape
            )
        )
        if (array_1_ndim == 1) and (array_2_ndim == 1):
            cardinality = len(set(array_1).union(set(array_2)))
            array_1_unique = set(array_1)
            array_2_unique = set(array_2)
            array_intersection = array_1_unique.intersection(array_2_unique)
            array_1_outer = array_1_unique - array_intersection
            array_2_outer = array_2_unique - array_intersection
            self.info("Compare {} vs {} - cardinality :")
            self.info(
                "Intersection {} and {} : {} ({:.2f}%)".format(
                    array_1_name,
                    array_2_name,
                    len(array_intersection),
                    len(array_intersection) / cardinality,
                )
            )
            self.info(
                "Unique Values in {} but not in {} : {} ({:.2f}%)".format(
                    array_1_name,
                    array_2_name,
                    len(array_1_outer),
                    len(array_1_outer) / cardinality,
                )
            )
            self.info(
                "Unique Values in {} but not in {} : {} ({:.2f}%)".format(
                    array_2_name,
                    array_1_name,
                    len(array_2_outer),
                    len(array_2_outer) / cardinality,
                )
            )

    def dict(self, dictionary, dictionary_name):
        n_values = len(dictionary)
        self.info("Dictionary {} length : {}".format(dictionary_name, n_values))
        self.info("Dictionary {}".format(dictionary_name))
        for key, value in dictionary.items():
            self.info('{} - {}'.format(key, value))

    def list(self, input_list, input_list_name):
        n_values = len(input_list)
        self.info("List {} length : {}".format(input_list_name, n_values))
        self.info('List {} : {}'.format(input_list_name, input_list))

    def set(self, input_set, input_set_name):
        n_values = len(input_set)
        self.info("Set {} length : {}".format(input_set_name, n_values))
        self.info("Set {} : {}".format(input_set_name, input_set))

    def savepath(self, file_to_save, save_path):
        self.info("Saving {} to path : {}".format(file_to_save, save_path))

    def loadpath(self, file_to_load, load_path):
        self.info("Loading {} from path : {}".format(file_to_load, load_path))

    def scikit(self, model, model_name):
        self.info("Model {} type : {}".format(model_name, type(model).__name__))
        self.dict(model.get_params(), "Parameters of scikit-learn model {}".format(model_name))

    def xgboost(self, model, model_name):
        raise NotImplementedError

    def lightgbm(self, model, model_name):
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
