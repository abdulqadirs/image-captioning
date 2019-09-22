import logging

LOG_LEVEL = logging.INFO
logger = logging.getLogger("captioning")

def setup_logging(logfile, level):
    global logger
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s', "%H:%M:%S")
    logger.setLevel(level)
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(level)
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    if logfile:
        filehandler = logging.FileHandler(logfile)
        filehandler.setLevel(level)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
