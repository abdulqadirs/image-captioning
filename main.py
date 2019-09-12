import configparser
from data import data_loaders


config = configparser.ConfigParser()
config.read('config.ini')
images_path = config['paths']['images_dir']
captions_path = config['paths']['captions_dir']

training_loader, validation_loader, testing_loader = data_loaders(images_path, captions_path)