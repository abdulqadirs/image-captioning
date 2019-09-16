import configparser
from data import data_loaders
from encoder import Encoder
from decoder import Decoder
from pretrained_embeddings import load_pretrained_embeddings
from data import read_captions, dictionary
from loss_function import criterion

config = configparser.ConfigParser()
config.read('config.ini')
images_path = config['paths']['images_dir']
captions_path = config['paths']['captions_dir']
embed_size = int(config['encoder']['embed_size'])
hidden_size = int(config['decoder']['hidden_size'])
batch_size = int(config['train']['batch_size'])
captions_path = config['paths']['captions_dir']
raw_captions = read_captions(captions_path)
id_to_word, word_to_id = dictionary(raw_captions, threshold = 5)
vocab_size = len(id_to_word)

training_loader, validation_loader, testing_loader = data_loaders(images_path, captions_path)

images, captions, lengths = next(iter(training_loader))
#pretrained_embeddings
pretrained_embeddings = load_pretrained_embeddings()
#encoder
encoder = Encoder(embed_size)
image_features = encoder(images)
#decoder
decoder = Decoder( embed_size, hidden_size, vocab_size, batch_size)
predicted_captions = decoder(image_features, captions, lengths, pretrained_embeddings)
print(predicted_captions.shape)

#loss function
loss = criterion(predicted_captions, captions)
print(loss)