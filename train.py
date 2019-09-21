import configparser
from data import data_loaders
from models.encoder import Encoder
from models.decoder import Decoder
from pretrained_embeddings import load_pretrained_embeddings
from data import read_captions, dictionary
from loss_function import criterion
import torch
import numpy as np
from utils.checkpoint import save_checkpoint


class ImageCaptioning:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.images_path = self.config['paths']['images_dir']
        self.captions_path = self.config['paths']['captions_dir']
        self.embed_size = int(self.config['encoder']['embed_size'])
        self.hidden_size = int(self.config['decoder']['hidden_size'])
        self.batch_size = int(self.config['train']['batch_size'])
        self.captions_path = self.config['paths']['captions_dir']
        self.epochs = int(self.config['train']['epochs'])
        self.raw_captions = read_captions(self.captions_path)
        self.id_to_word, self.word_to_id = dictionary(self.raw_captions, threshold = 5)
        self.vocab_size = len(self.id_to_word)
        self.learning_rate = float(self.config['train']['learning_rate'])
        self.training_loader, self.validation_loader, self.testing_loader = data_loaders(self.images_path, self.captions_path)

        #images, captions, lengths = next(iter(training_loader))
        #pretrained_embeddings
        self.pretrained_embeddings = load_pretrained_embeddings()
        #encoder
        self.encoder = Encoder(self.embed_size)
        #decoder
        self.decoder = Decoder(self.embed_size, self.hidden_size, self.vocab_size, self.batch_size)

        #optimizer
        self.params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.learning_rate)
        self.validate_every = int(self.config['validation']['validate_every'])
        self.outdir = self.config['train']['outdir']
    def train(self):
        """
        training to generate captions for given images
        """
        for epoch in range(self.epochs):
            training_batch_losses = []
            for i, data in enumerate(self.training_loader, 0):
                images, captions, lengths = data
                self.optimizer.zero_grad()
                #image features
                image_features = self.encoder(images)
                #predicted captions
                predicted_captions = self.decoder(image_features, captions, lengths, self.pretrained_embeddings)
                #loss function
                loss = criterion(predicted_captions, captions)
                #calculating the gradients
                loss.backward()
                #updating the parameters
                self.optimizer.step()
                training_batch_losses.append(loss.item())
                print(loss.item())
                # mean loss per epoch
                #print("epoch: ", epoch)
                #print(np.mean(training_batch_losses))
            if epoch % self.validate_every == 0:
                self.validation()
                save_checkpoint(epoch = epoch,
                                outdir = self.outdir,
                                encoder = self.encoder,
                                decoder = self.decoder,
                                optimizer = self.optimizer,
                                criterion = criterion)


    
    def validation(self):
        for _, data in enumerate(self.validation_loader, 0):
            images, captions, lengths = data
            #image features
            image_features = self.encoder(images)
            #predicted captions
            predicted_captions = self.decoder(image_features, captions, lengths, self.pretrained_embeddings)
            #loss function
            loss = criterion(predicted_captions, captions)

        print('validation error: ', loss)    

    def testing(self):
        for _, data in enumerate(self.testing_loader, 0):
            images, captions, lengths = data
            #image features
            image_features = self.encoder(images)
            #predicted captions
            predicted_captions = self.decoder(image_features, captions, lengths, self.pretrained_embeddings)
            #loss function
            loss = criterion(predicted_captions, captions)
        
        return loss

image_captioning = ImageCaptioning()
image_captioning.train()