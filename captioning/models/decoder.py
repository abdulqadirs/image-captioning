import torch
import torch.nn as nn

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
        # embedding layer
        #self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        
        # Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.no_lstm_units, num_layers = self.num_layers,
                           batch_first = True)
    
        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.no_lstm_units, out_features=self.vocab_size)
    
        # activations
        self.softmax = nn.Softmax(dim=0)
  
    def forward(self, features, captions, lenghts, pretrained_embeddings):
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
        #embed = nn.Embedding.from_pretrained(pretrained_embeddings)
        embeddings = pretrained_embeddings(captions)
        #print(embeddings.shape)
        #print('features unsqueeze', features.unsqueeze(1).shape)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        #print('embeddings shape after', embeddings.shape)
        #packed = pack_padded_sequence(embeddings, lengths, batch_first = True)
        #features = features.unsqueeze(1)
        #print(features)
        #features.view(2, 20, self.embed_dim)
        lstm_output, _ = self.lstm(embeddings)
        #hiddens, _ = pad_packed_sequence(packed, batch_first=True)
        lstm_output_reshaped = lstm_output[:,1:,:]
        outputs = self.fc_out(lstm_output_reshaped)
        #outputs = self.fc_out(lstm_output)
        # batch_size, captions_length, vocab_length = outputs.size()
        # for i in range(batch_size):
        #     for j in range(captions_length):
        #         outputs[i][j] = self.softmax(outputs[i][j])

        
        return outputs
            
