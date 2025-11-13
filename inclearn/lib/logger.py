import logging
import datetime


def set_logging_level(logging_level,log_file):
    logging_level = logging_level.lower()

    if logging_level == "critical":
        level = logging.CRITICAL
    elif logging_level == "warning":
        level = logging.WARNING
    elif logging_level == "info":
        level = logging.INFO
    else:
        level = logging.DEBUG
    #log_file = "logfile_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        format='%(asctime)s [%(filename)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=level
    )