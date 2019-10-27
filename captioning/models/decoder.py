#https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class Decoder(nn.Module):
    """
    Decodes the image embeddings to captions using LSTM.

    Attributes:
        embed_size (int): Size of embedded image.
        hidden_size (int): No of cells in LSTM.
        vocab_size (int): Size of vocabulary.
        batch_size (int): Batch size.
        num_layers (int): No of layers in LSTM.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, batch_size, num_layers = 1):
        super(Decoder, self).__init__()
        
        self.embed_dim = embed_size
        self.no_lstm_units = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_length=20
        self.batch_size = batch_size
        
        # Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.no_lstm_units, num_layers = self.num_layers,
                           batch_first = True)
    
        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.no_lstm_units, out_features=self.vocab_size)
    
        # activations
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, features, lengths, pretrained_embeddings):
        """
        Decodes the encoded image features to captions without teacher foring.

        Args:
            features (tensor): Encoded image features.
            lengths (tensor): Actual lengths of captions.
            pretrained_embeddings (): Word2vec embeddings of words of vocabulary.
        
        Returns: 
            predicted_captions (tensor): Predicted captions of shape(batch_size, captions_length)
            predicted_captions_prob (tensor): probability of predicted captiosn (batch_size, captions_length)
        """
        predicted_caption_ids= []
        predicted_caption_prob = []
        inputs = features.unsqueeze(1)
        #inputs.shape: (batch_size, 1, encoder_ouput_dim)
        max_seq_length, _ = lengths.max(0)
        states = None
        for i in range(max_seq_length):
            #input = (batch_size, 1, pretrained_emb_dim=encoder_output_dim) 
            hiddens, states = self.lstm(inputs, states)
            #hiddens = (batch_size, 1, no_of_lstm_units)
            outputs = self.fc_out(hiddens)
            #outputs = (batch_size, 1, vocab_size)
            outputs = outputs.squeeze(1)
            #outputs = (batch_size, vocab_size)
            predicted_values, predicted_indices = outputs.max(1)
            #predicted = (batch_size)
            predicted_caption_ids.append(predicted_indices)
            inputs = pretrained_embeddings(predicted_indices)
            #inputs = (batch_size, pretrained_emb_dim)
            inputs = inputs.unsqueeze(1)
            #inputs = (batch_size, 1, pretrained_emb_dim)
            predicted_caption_prob.append(predicted_values)
        predicted_caption_ids = torch.stack(predicted_caption_ids, 1)
        predicted_caption_prob = torch.stack(predicted_caption_prob, 1)
        #predicted_captions = (batch_size, max_seq_length)
      
        return predicted_caption_ids, predicted_caption_prob
  
    def teacher_forcing(self, features, captions, lengths, pretrained_embeddings):
        """
        Decodes the encoded image features to captions with teacher forcing.

        Args:
            features (tensor): Encoded image features.
            captions (tensor): Reference captions.
            lengths (tensor): Actual lengths of captions.
            pretrained_embeddings (): Word2vec embeddings of words of vocabulary.
        
        Returns: 
            output (tensor): Predicted captions of shape(batch_size, captions_length, vocab_length)
        """
        #initializing the captions with pretrained embeddings.
        #captions.shape: 1d torch tensor
        embeddings = pretrained_embeddings(captions)
        #embeddings.shape: (batch_size, max_caption_len_in_batch, pretrained_emb_dim)

        #teacher forcing
        #concatenating the image features with captions' embeddigns.
        #features.shape: (batch_size, decoder_output_dim) 
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        #embeddings.shape: (batch_size, max_caption_len_in_batch + 1, pretrained_emb_dim)

        #pack padded sequence
        #packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first = True)

        #pass the packed embeddings through LSTM.
        lstm_output, _ = self.lstm(embeddings)
        #lstm_output.shape: (batch_size, max_caption_len_in_batch + 1, no_of_lstm_units)

        #pad the packed embeddings(output of lstm)
        #padded_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        #removing the first output
        lstm_output_reshaped = lstm_output[:,1:,:]
        #outputs.shape: (batch_size, max_caption_len_in_batch, no_of_lstm_units)

        #linear transformation
        outputs = self.fc_out(lstm_output_reshaped)
        #outptus: (batch_size, max_caption_len_in_batch, vocab_size)

        #softmax
        # batch_size, captions_length, vocab_length = outputs.size()
        # for i in range(batch_size):
        #     for j in range(captions_length):
        #         outputs[i][j] = self.softmax(outputs[i][j])

        
        return outputs