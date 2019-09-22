import configparser
import logging
from config import Config

logger = logging.getLogger('captioning')

def read_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    logger.info("Reading the config file from: %s" % file_path)

    #GPUs

    #paths
    Config.set("images_dir", config.get("paths", "images_dir"))
    Config.set("captions_dir", config.get("paths", "captions_dir"))
    Config.set("output_dir", config.get("paths", "outdir"))
    Config.set("checkpoint_file", config.get("paths", "checkpoint_file"))

    #pretrained_embeddings
    Config.set("pretrained_emb_path", config.get("pretrained_embeddings", "path"))
    Config.set("pretrained_emb_dim", config.getint("pretrained_embeddings", "emb_dim"))

    #encoder
    Config.set("encoder_embed_size", config.getint("encoder", "embed_size"))

    #decoder
    Config.set("decoder_embed_size", config.getint("decoder", "embed_size"))
    Config.set("decoder_hidden_size", config.getint("decoder", "hidden_size"))

    #Training
    Config.set("training_batch_size", config.getint("training_batch_size", "batch_size"))
    Config.set("epochs", config.getint("training", "epochs"))
    Config.set("learning_rate", config.getfloat("training", "learning_rate"))

    #validation
    Config.set("validation_batch_size", config.getint("validation", "batch_size"))
    Config.set("validate_every", config.getint("validation", "validate_every"))

    #testing


    #logging
    Config.set("logfile", config.get("logging", "logifile"))



