import logging
from pathlib import Path
from models.encoder import Encoder
from models.decoder import Decoder
from image_captioning import ImageCaptioning
from utils.setup_logging import setup_logging
import configparser
from data import data_loaders, read_captions, dictionary
from pretrained_embeddings import load_pretrained_embeddings
from optimizer import adam_optimizer
from utils.checkpoint import load_checkpoint
from config import Config
from utils.read_config import reading_config
from loss_function import cross_entropy
from torch.utils.tensorboard import SummaryWriter
from utils.parse_arguments import parse_arguments

logger = logging.getLogger('captioning')

def main():

    #parsing the arguments
    #args = parse_arguments()
   
    #setup logging
    #output_dir = Path('/content/drive/My Drive/image-captioning/output')
    output_dir = Path('output')
    #output_dir = Path(args.output_directory)
    logfile_path = output_dir / "output.log"
    setup_logging(logfile_path, logging.INFO)

    #setup and read config.ini
    config_file = '../config.ini'
    reading_config(config_file)

    #tensorboard
    tensorboard_writer = SummaryWriter(output_dir / 'train')
    
    #load dataset
    #dataset_dir = Path('/content/drive/My Drive/Flickr8k_Dataset')
    dataset_dir = Path('../../Flickr8k_Dataset')
    #dataset_dir = Path(args.dataset)
    images_path = Config.get("images_dir")
    captions_path = Config.get("captions_dir")
    training_loader, validation_loader, testing_loader = data_loaders(dataset_dir / images_path,
                                                                     dataset_dir / captions_path)

    #load pretrained embeddings
    #pretrained_emb_dir = Path('/content/drive/My Drive/word2vec')
    pretrained_emb_dir = Path('../../word2vec')
    #pretrained_emb_dir = Path(args.pretrained_embeddings)
    pretrained_emb_file = Config.get("pretrained_emb_path")
    pretrained_embeddings = load_pretrained_embeddings(pretrained_emb_dir / pretrained_emb_file, 
                                                        dataset_dir / captions_path)

    #load the model (encoder, decoder, optimizer)
    embed_size = Config.get("encoder_embed_size")
    hidden_size = Config.get("decoder_hidden_size")
    batch_size = Config.get("training_batch_size")
    epochs = Config.get("epochs")
    raw_captions = read_captions(dataset_dir / captions_path)
    id_to_word, word_to_id = dictionary(raw_captions, threshold = 5)
    vocab_size = len(id_to_word)
    encoder = Encoder(embed_size)
    decoder = Decoder(embed_size, hidden_size, vocab_size, batch_size)
    
    #load the optimizer
    learning_rate = Config.get("learning_rate")
    optimizer = adam_optimizer(encoder, decoder, learning_rate)

    #loss funciton
    criterion = cross_entropy

    #load checkpoint
    checkpoint_file = Config.get("checkpoint_file")
    checkpoint_captioning = load_checkpoint(output_dir / checkpoint_file)

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
    
    #training
    validate_every = Config.get("validate_every")
    #model.train(epochs, validate_every, start_epoch)

    #testing
    model.testing(id_to_word, dataset_dir / images_path / images_path)



if __name__ == "__main__":
    main()
