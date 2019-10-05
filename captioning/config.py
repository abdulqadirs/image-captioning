import logging
from copy import deepcopy

logger = logging.getLogger('captioning')

class Config:
    """
    stores the content of config.ini using deepcopy(without changing the source)
    """
    __config = {}

    @staticmethod
    def get(name):
        try:
            return deepcopy(Config.__config[name])
        except KeyError:
            logger.warning("Config setting " + name + "not found.")
            return
    
    @staticmethod
    def set(name, value):
        try:
            Config.__config[name] = deepcopy(value)
        except:
            Config.__config[name] = value