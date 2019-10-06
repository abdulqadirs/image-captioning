import logging

LOG_LEVEL = logging.INFO
logger = logging.getLogger("captioning")

def setup_logging(logfile, level):
    """
    Sets up logging to stout and to the given file

    Args:
        logfile (Path): The path of logging file
        level: The level of logging(logging.info)
    
    Raises:
        InvalidPath: An error if the given file path is invalid.
    """
    global logger
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s', "%H:%M:%S")
    logger.setLevel(level)
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(level)
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    # TODO (aq): Raise error if the file path is invalid or file doesn't exist.
    filehandler = logging.FileHandler(logfile)
    filehandler.setLevel(level)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
