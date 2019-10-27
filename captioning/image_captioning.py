import torch
import numpy as np
import logging
from tqdm import tqdm

from utils.checkpoint import save_checkpoint
from config import Config
from utils.stats import Statistics
from metrics.bleu import bleu_score
from decode_caption.detokenize import detokenize_caption
from decode_caption.greedy_search import greedy_search
from decode_caption.save_captions import save_captions

logger = logging.getLogger("captioning")

class ImageCaptioning:
    """
    Runs the model on training, validation and test datasets.

    Attribute:
        encoder (object): Images encoder.
        decoder (object): Decodes the encoded images to captions.
        optimizer (object): Gradient descent optimizer.
        criterion (object): Loss function.
        training_loader:
        validation_loader:
        testing_loader:
        pretrained_embeddings:
        output_dir (Path):
        tensorboard_writer (object):
    """
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
        Runs the model on training dataset.
        
        Args:
            epochs (int): Total epochs.
            validate_every (int): Run validation after every validate_every no of epochs.
            start_epoch (int): Starting epoch if using the stored checkpoint.
        """
        #self.validation(epoch = 0)
        #batch_size = Config.get("training_batch_size")
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
                predicted_captions = self.decoder.teacher_forcing(image_features, captions, lengths, self.pretrained_embeddings)
                #max_length, _ = lengths.max(0)
                #ref_captions_mask = torch.ones(batch_size, max_length).to(Config.get("device"))
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
                self.validation(epoch = epoch)
                save_checkpoint(epoch = epoch,
                                outdir = self.output_dir,
                                encoder = self.encoder,
                                decoder = self.decoder,
                                optimizer = self.optimizer,
                                criterion = self.criterion)


    
    def validation(self, epoch = None):
        """
        Runs the model on validation dataset.

        Args:
            epoch (int): Current epoch.
        """
        batch_size = Config.get("validation_batch_size")
        #validation_losses = []
        raw_bleu1 = []
        raw_bleu2 = []
        raw_bleu3 = []
        raw_bleu4 = []
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
            predicted_captions, predicted_captions_prob = self.decoder(image_features, lengths, self.pretrained_embeddings)
            predicted_captions = predicted_captions.data.cpu()
            predicted_captions_prob = predicted_captions_prob.data.cpu()
            predicted_captions = predicted_captions.squeeze(0).tolist()
    
            captions = captions.data.cpu()
            #max_length, _ = lengths.max(0)
            #ref_captions_mask = torch.ones(batch_size, max_length).data.cpu()
             #loss function
            #loss = self.criterion(predicted_captions_prob, ref_captions_mask)
            #validation_losses.append(loss)
            #calculating the bleu score
            #tokenized_captions = greedy_search(predicted_captions)
            bleu1, bleu2, bleu3, bleu4 = bleu_score(captions, predicted_captions)
            raw_bleu1.append(bleu1)
            raw_bleu2.append(bleu2)
            raw_bleu3.append(bleu3)
            raw_bleu4.append(bleu4)

        self.stat.record(bleu1=np.mean(raw_bleu1))
        self.stat.record(bleu2=np.mean(raw_bleu2))
        self.stat.record(bleu3=np.mean(raw_bleu3))
        self.stat.record(bleu4=np.mean(raw_bleu4))
        #self.stat.record(validation_losses=np.mean(validation_losses))
        #self.stat.push_tensorboard_losses(epoch)
        self.stat.push_tensorboard_eval(epoch, "validation")
        self.stat.log_eval(epoch, "validation")


    def testing(self, id_to_word, images_dir):
        """
        Runs the model on test dataset.

        Args:
            id_to_word (list):
            images_dir (Path): Path of images dataset directory.
        """
        raw_bleu1 = []
        raw_bleu2 = []
        raw_bleu3 = []
        raw_bleu4 = []
        for _, data in tqdm(enumerate(self.testing_loader, 0)):
            images, captions, lengths, image_ids = data
            images = images.to(Config.get("device"))
            captions = captions.to(Config.get("device"))
            #image features
            image_features = self.encoder(images)
            #predicted captions
            predicted_captions, predicted_captions_prob = self.decoder(image_features, lengths, self.pretrained_embeddings)
            predicted_captions = predicted_captions.data.cpu()
            predicted_captions_prob = predicted_captions_prob.data.cpu()
            #loss function
            #loss = self.criterion(predicted_captions, captions)
            #calculating the bleu score
            predicted_captions = predicted_captions.squeeze(0).tolist()
            bleu1, bleu2, bleu3, bleu4 = bleu_score(captions, predicted_captions)
            raw_bleu1.append(bleu1)
            raw_bleu2.append(bleu2)
            raw_bleu3.append(bleu3)
            raw_bleu4.append(bleu4)

            #decoding the predicted captions and saving them
            detokenized_caption = detokenize_caption(predicted_captions, id_to_word)
            save_captions(images_dir, self.output_dir, image_ids, detokenized_caption)
        
        self.stat.record(bleu1=np.mean(raw_bleu1))
        self.stat.record(bleu2=np.mean(raw_bleu2))
        self.stat.record(bleu3=np.mean(raw_bleu3))
        self.stat.record(bleu4=np.mean(raw_bleu4))
        epoch = 1
        self.stat.push_tensorboard_eval(epoch, "testing")
        self.stat.log_eval(epoch, "testing")
