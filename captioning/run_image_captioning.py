import logging
from pathlib import Path
import configparser
from torch.utils.tensorboard import SummaryWriter

from utils.setup_logging import setup_logging
from loss_function import cross_entropy
from models.encoder import Encoder
from models.decoder import Decoder
from image_captioning import ImageCaptioning
from data import data_loaders, read_captions, dictionary
from pretrained_embeddings import load_pretrained_embeddings
from optimizer import adam_optimizer
from utils.checkpoint import load_checkpoint
from config import Config
from utils.read_config import reading_config
from utils.parse_arguments import parse_arguments

logger = logging.getLogger('captioning')

def main():

    #parsing the arguments
    args, _ = parse_arguments()
   
    #setup logging
    #output_dir = Path('/content/drive/My Drive/image-captioning/output')
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    logfile_path = Path(output_dir / "output.log")
    setup_logging(logfile = logfile_path)

    #setup and read config.ini
    #config_file = Path('/content/drive/My Drive/image-captioning/config.ini')
    config_file = Path('../config.ini')
    reading_config(config_file)

    #tensorboard
    tensorboard_logfile = Path(output_dir / 'tensorboard')
    tensorboard_writer = SummaryWriter(tensorboard_logfile)
    
    #load dataset
    #dataset_dir = Path('/content/drive/My Drive/Flickr8k_Dataset')
    dataset_dir = Path(args.dataset)
    images_path = Path(dataset_dir / Config.get("images_dir"))
    captions_path = Path(dataset_dir / Config.get("captions_dir"))
    training_loader, validation_loader, testing_loader = data_loaders(images_path, captions_path)


    #load the model (encoder, decoder, optimizer)
    embed_size = Config.get("encoder_embed_size")
    hidden_size = Config.get("decoder_hidden_size")
    batch_size = Config.get("training_batch_size")
    epochs = Config.get("epochs")
    raw_captions = read_captions(captions_path)
    id_to_word, word_to_id = dictionary(raw_captions, threshold = 5)
    vocab_size = len(id_to_word)
    encoder = Encoder(embed_size)
    decoder = Decoder(embed_size, hidden_size, vocab_size, batch_size)

    #load pretrained embeddings
    #pretrained_emb_dir = Path('/content/drive/My Drive/word2vec')
    pretrained_emb_dir = Path(args.pretrained_embeddings)
    pretrained_emb_file = Path(pretrained_emb_dir / Config.get("pretrained_emb_path"))
    pretrained_embeddings = load_pretrained_embeddings(pretrained_emb_file, id_to_word)
    
    #load the optimizer
    learning_rate = Config.get("learning_rate")
    optimizer = adam_optimizer(encoder, decoder, learning_rate)

    #loss funciton
    criterion = cross_entropy

    #load checkpoint
    checkpoint_file = Path(output_dir / Config.get("checkpoint_file"))
    checkpoint_captioning = load_checkpoint(checkpoint_file)

    #using available device(gpu/cpu)
    encoder = encoder.to(Config.get("device"))
    decoder = decoder.to(Config.get("device"))
    pretrained_embeddings = pretrained_embeddings.to(Config.get("device"))
        

    start_epoch = 1
    if checkpoint_captioning is not None:
        start_epoch = checkpoint_captioning['epoch'] + 1
        encoder.load_state_dict(checkpoint_captioning['encoder'])
        decoder.load_state_dict(checkpoint_captioning['decoder'])
        optimizer.load_state_dict(checkpoint_captioning['optimizer'])
        logger.info('Initialized encoder, decoder and optimizer from loaded checkpoint')

    del checkpoint_captioning

    #image captioning model
    model = ImageCaptioning(encoder, decoder, optimizer, criterion,
                                            training_loader, validation_loader, testing_loader,
                                            pretrained_embeddings, output_dir, tensorboard_writer)
    
    #training and testing the model
    if args.training:
        validate_every = Config.get("validate_every")
        model.train(epochs, validate_every, start_epoch)
    elif args.testing:
        images_path = Path(images_path / Config.get("images_dir"))
        model.testing(id_to_word, images_path)




if __name__ == "__main__":
    main()