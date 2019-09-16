from gensim import models
import configparser
from data import read_captions, dictionary
import torch
import numpy as np
import torch.nn as nn


def load_pretrained_embeddings():
    config = configparser.ConfigParser()
    config.read('config.ini')
    pretrained_emb_path = config['pretrained_embeddings']['path']
    captions_path = config['paths']['captions_dir']
    emb_dim = int(config['pretrained_embeddings']['emb_dim'])
    model = models.KeyedVectors.load_word2vec_format(pretrained_emb_path, binary=False)
    raw_captions = read_captions(captions_path)
    id_to_word, word_to_id = dictionary(raw_captions, threshold = 5)
    vocab = id_to_word
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    bias = np.sqrt(3.0 / embeddings.size(1))
    pretrained_embeddings = torch.nn.init.uniform_(embeddings, -bias, bias)
    for i, word in enumerate(vocab):
        try:
            vector = model.word_vec(word.lower())
            pretrained_embeddings[i] = torch.FloatTensor(vector)
        except:
            continue

    embeddings = nn.Embedding.from_pretrained(pretrained_embeddings)

    return embeddings
