import logging
import sys


class LoggingConfig:

    def __init__(self, level):
        level_config = LoggingConfig.get_level_config()

        if level not in level_config:
            sys.exit('Unexisting log level!')

        logging.basicConfig(level=level_config[level])

    @staticmethod
    def get_level_config():
        return {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
