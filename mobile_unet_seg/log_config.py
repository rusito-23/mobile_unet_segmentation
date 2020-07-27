"""
Logging Configuration.
- log to file -> level: DEBUG
- log to console -> level: INFO
"""
import logging
import sys


def config_logger(log_file):

    logFormatter = logging.Formatter('%(asctime)s '
                                     '[%(levelname)s] '
                                     '%(message)s')
    rootLogger = logging.getLogger('mobile_unet_seg')

    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.DEBUG)
