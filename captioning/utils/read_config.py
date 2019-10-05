import configparser
import logging
from config import Config
import torch

logger = logging.getLogger('captioning')

def reading_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    logger.info('Reading the config file from: %s' % file_path)

    #GPUs
    Config.set("disable_cuda", config.getboolean("GPU", "disable_cuda", fallback=False))
    if not Config.get("disable_cuda") and torch.cuda.is_available():
        Config.set("device", "cuda")
        logger.info('GPU is available')
    else:
        Config.set("device", "cpu")
        logger.info('Only CPU is available')

    #paths
    Config.set("images_dir", config.get("paths", "images_dir", fallback="Flickr8k_images"))
    Config.set("captions_dir", config.get("paths", "captions_dir", fallback="Flickr8k_text/Flickr8k.token.txt"))
    Config.set("output_dir", config.get("paths", "outdir", fallback="output"))
    Config.set("checkpoint_file", config.get("paths", "checkpoint_file", fallback="checkpoint.ImageCaptioning.pth.tar"))

    #pretrained_embeddings
    Config.set("pretrained_emb_path", config.get("pretrained_embeddings", "path", fallback="glove.6B.50d.word2vec.txt"))
    Config.set("pretrained_emb_dim", config.getint("pretrained_embeddings", "emb_dim", fallback=50))

    #encoder
    Config.set("encoder_embed_size", config.getint("encoder", "embed_size", fallback=50))

    #decoder
    Config.set("decoder_embed_size", config.getint("decoder", "embed_size", fallback=50))
    Config.set("decoder_hidden_size", config.getint("decoder", "hidden_size", fallback=20))

    #Training
    Config.set("training_batch_size", config.getint("training", "batch_size", fallback=2))
    Config.set("epochs", config.getint("training", "epochs", fallback=1000))
    Config.set("learning_rate", config.getfloat("training", "learning_rate", fallback=0.01))

    #validation
    Config.set("validation_batch_size", config.getint("validation", "batch_size", fallback=2))
    Config.set("validate_every", config.getint("validation", "validate_every", fallback=5))

    #testing


    #logging
    Config.set("logfile", config.get("logging", "logfile", fallback="output.log"))


