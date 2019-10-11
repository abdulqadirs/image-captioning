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
  
    def forward(self, features, captions, lengths, pretrained_embeddings):
        """
        Decodes the encoded image features to captions.

        Args:
            features (tensor): Encoded image features.
            captions (tensor): Reference captions.
            lengths (tensor): Actual lengths of captions.
            pretrained_embeddings (): Word2vec embeddings of words of vocabulary.
        
        Returns: 
            output (tensor): Predicted captions of shape(batch_size, captions_length, vocab_length)
        """
        #initializing the captions with pretrained embeddings.
        embeddings = pretrained_embeddings(captions)
        #concatenating the image features with captions' embeddigns.
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        #pack padded sequence
        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first = True)
        #pass the packed embeddings through LSTM.
        packed_output, _ = self.lstm(packed_embeddings)
        #pad the packed embeddings(output of lstm)
        padded_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        #lstm_output_reshaped = lstm_output[:,1:,:]
        outputs = self.fc_out(padded_output)
        #outputs = self.fc_out(lstm_output)
        # batch_size, captions_length, vocab_length = outputs.size()
        # for i in range(batch_size):
        #     for j in range(captions_length):
        #         outputs[i][j] = self.softmax(outputs[i][j])

        
        return outputs
            
