import configparser
from data import data_loaders
from encoder import Encoder
from decoder import Decoder
from pretrained_embeddings import load_pretrained_embeddings
from data import read_captions, dictionary
from loss_function import criterion
import torch


config = configparser.ConfigParser()
config.read('config.ini')
images_path = config['paths']['images_dir']
captions_path = config['paths']['captions_dir']
embed_size = int(config['encoder']['embed_size'])
hidden_size = int(config['decoder']['hidden_size'])
batch_size = int(config['train']['batch_size'])
captions_path = config['paths']['captions_dir']
epochs = int(config['train']['epochs'])
raw_captions = read_captions(captions_path)
id_to_word, word_to_id = dictionary(raw_captions, threshold = 5)
vocab_size = len(id_to_word)
learning_rate = float(config['train']['learning_rate'])
training_loader, validation_loader, testing_loader = data_loaders(images_path, captions_path)

#images, captions, lengths = next(iter(training_loader))
#pretrained_embeddings
pretrained_embeddings = load_pretrained_embeddings()
#encoder
encoder = Encoder(embed_size)
#decoder
decoder = Decoder( embed_size, hidden_size, vocab_size, batch_size)

#optimizer
params = list(decoder.parameters()) + list(encoder.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)


for epoch in range(epochs):
    training_batch_losses = []
    for i, data in enumerate(training_loader, 0):
        images, captions, lengths = data
        optimizer.zero_grad()
        #image features
        image_features = encoder(images)
        #predicted captions
        predicted_captions = decoder(image_features, captions, lengths, pretrained_embeddings)
        #loss function
        loss = criterion(predicted_captions, captions)
        #calculating the gradients
        loss.backward()
        #updating the parameters
        optimizer.step()

        print(epoch)

