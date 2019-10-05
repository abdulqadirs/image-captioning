import torch
import numpy as np
from utils.checkpoint import save_checkpoint
from config import Config
import logging
from utils.stats import Statistics
from tqdm import tqdm
from metrics.bleu import bleu_score
from decode_caption.detokenize import detokenize_caption
from decode_caption.greedy_search import greedy_search
from decode_caption.save_captions import save_captions

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

    def train(self,epochs, validate_every, start_epoch):
        """
        training to generate captions for given images
        """
        for epoch in range(start_epoch, epochs + 1):
            training_batch_losses = []
            for _, data in tqdm(enumerate(self.training_loader, 0)):
                images, captions, lengths, _ = data
                self.optimizer.zero_grad()
                images = images.to(Config.get("device"))
                captions = captions.to(Config.get("device"))
                #setting up training mode
                self.encoder = self.encoder.train()
                self.decoder = self.decoder.train()
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

            self.stat.record(training_losses=np.mean(training_batch_losses))
            self.stat.push_tensorboard_losses(epoch)
            self.stat.log_losses(epoch)
            if (epoch -1) % validate_every == 0:

                self.validation(epoch)
                save_checkpoint(epoch = epoch,
                                outdir = self.output_dir,
                                encoder = self.encoder,
                                decoder = self.decoder,
                                optimizer = self.optimizer,
                                criterion = self.criterion)


    
    def validation(self, epoch):
        """
        validation
        """
        validation_losses = []
        raw_bleu_score = []
        for _, data in tqdm(enumerate(self.validation_loader, 0)):
            images, captions, lengths, _ = data
            images = images.to(Config.get("device"))
            captions = captions.to(Config.get("device"))
            #setting evaluation mode
            self.encoder.eval()
            self.decoder.eval()
            #image features
            image_features = self.encoder(images)
            #predicted captions
            predicted_captions = self.decoder(image_features, captions, lengths, self.pretrained_embeddings).data.cpu()
            #loss function
            captions = captions.data.cpu()
            loss = self.criterion(predicted_captions, captions)
            validation_losses.append(loss)
            #calculating the bleu score
            tokenized_captions = greedy_search(predicted_captions)
            bleu = bleu_score(captions, tokenized_captions)
            raw_bleu_score.append(bleu)

        self.stat.record(bleu_score=np.mean(raw_bleu_score))
        self.stat.record(validation_losses=np.mean(validation_losses))
        self.stat.push_tensorboard_losses(epoch)
        self.stat.push_tensorboard_eval(epoch, "validation")
        self.stat.log_eval(epoch, "validation")


    def testing(self, id_to_word, images_dir):
        """
        testing
        """
        raw_bleu_score = []
        for _, data in tqdm(enumerate(self.testing_loader, 0)):
            images, captions, lengths, image_ids = data
            images = images.to(Config.get("device"))
            captions = captions.to(Config.get("device"))
            #image features
            image_features = self.encoder(images)
            #predicted captions
            predicted_captions = self.decoder(image_features, captions, lengths, self.pretrained_embeddings).data.cpu()
            #loss function
            loss = self.criterion(predicted_captions, captions)
            #calculating the bleu score
            tokenized_caption = greedy_search(predicted_captions)
            bleu = bleu_score(captions, tokenized_caption)
            raw_bleu_score.append(bleu)

            #decoding the predicted captions and saving them
            detokenized_caption = detokenize_caption(tokenized_caption, id_to_word)
            save_captions(images_dir, self.output_dir, image_ids, detokenized_caption)
        
        self.stat.record(bleu_score=np.mean(raw_bleu_score))
        epoch = 1
        self.stat.push_tensorboard_eval(epoch, "testing")
        self.stat.log_eval(epoch, "testing")
