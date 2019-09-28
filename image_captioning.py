import torch
import numpy as np
from utils.checkpoint import save_checkpoint
from config import Config
import logging
from utils.stats import Statistics

logger = logging.getLogger("captioning")

class ImageCaptioning:
    def __init__(self, encoder, decoder, optimizer, criterion, 
                traing_loader, validation_loader, testing_loader,
                 pretrained_embeddings, output_dir, tensorboard_writer):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_loader = traing_loader
        self.validation_loader = validation_loader
        self.testing_loader = testing_loader
        self.pretrained_embeddings = pretrained_embeddings
        self.output_dir = output_dir
        self.tensorboard_writer = tensorboard_writer
        self.stat = Statistics(output_dir, tensorboard_writer)

        self.encoder = self.encoder.to(Config.get("device"))
        self.decoder = self.decoder.to(Config.get("device"))
        self.pretrained_embeddings = self.pretrained_embeddings.to(Config.get("device"))

    def train(self,epochs, validate_every, start_epoch):
        """
        training to generate captions for given images
        """
        for epoch in range(start_epoch, epochs + 1):
            training_batch_losses = []
            for i, data in enumerate(self.training_loader, 0):
                images, captions, lengths = data
                self.optimizer.zero_grad()
                images = images.to(Config.get("device"))
                captions = captions.to(Config.get("device"))
                #image features
                image_features = self.encoder(images)
                #predicted captions
                predicted_captions = self.decoder(image_features, captions, lengths, self.pretrained_embeddings)
                #loss function
                loss = self.criterion(predicted_captions, captions)
                #calculating the gradients
                loss.backward()
                #updating the parameters
                self.optimizer.step()
                training_batch_losses.append(loss.item())
                print(loss.item())
                # mean loss per epoch
                #print("epoch: ", epoch)
                #print(np.mean(training_batch_losses))
                self.stat.record(training_losses=np.mean(training_batch_losses))
                self.stat.push_tensorboard_losses(epoch)
            #if (epoch -1) % validate_every == 0:
             #   self.validation()
                if i % 5 == 0:
                    save_checkpoint(epoch = epoch,
                                    outdir = self.output_dir,
                                    encoder = self.encoder,
                                    decoder = self.decoder,
                                    optimizer = self.optimizer,
                                    criterion = self.criterion)


    
    def validation(self):
        for _, data in enumerate(self.validation_loader, 0):
            images, captions, lengths = data
            images = images.to(Config.get("device"))
            captions = captions.to(Config.get("device"))
            #image features
            image_features = self.encoder(images)
            #predicted captions
            predicted_captions = self.decoder(image_features, captions, lengths, self.pretrained_embeddings)
            #loss function
            loss = self.criterion(predicted_captions, captions)

        print('validation error: ', loss)    

    def testing(self):
        for _, data in enumerate(self.testing_loader, 0):
            images, captions, lengths = data
            images = images.to(Config.get("device"))
            captions = captions.to(Config.get("device"))
            #image features
            image_features = self.encoder(images)
            #predicted captions
            predicted_captions = self.decoder(image_features, captions, lengths, self.pretrained_embeddings)
            #loss function
            loss = self.criterion(predicted_captions, captions)
        
        return loss

