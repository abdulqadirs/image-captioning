from gensim.models import KeyedVectors
from data import read_captions, dictionary
import torch
import numpy as np
import torch.nn as nn
from config import Config
import logging

logger = logging.getLogger("captioning")

def load_pretrained_embeddings(pretrained_emb_path, captions_path):
    """
    loads the pretrained word2vec and initialzes the words from vocabulary using the embeddings

    Params
    ------
    - prtrained_emb_path: path of the word2vec file

    Return
    ------
    - words of vocabulary initialized from pretrained word2vec embeddings
    """
    logger.info("Loading the pretrained embeddings ...")
    emb_dim = Config.get("pretrained_emb_dim")
    #laoding pretrained embeddings
    word_vectors = KeyedVectors.load_word2vec_format(pretrained_emb_path, binary=False)
    #reading the captions
    raw_captions = read_captions(captions_path)
    #loading dictionary
    id_to_word, word_to_id = dictionary(raw_captions, threshold = 5)
    vocab = id_to_word
    #initializing the embeddings randomly 
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    bias = np.sqrt(3.0 / embeddings.size(1))
    pretrained_embeddings = torch.nn.init.uniform_(embeddings, -bias, bias)
    #initializing the embeddings using pretrained word2vec embeddings
    for i, word in enumerate(vocab):
        try:
            vector = word_vectors.word_vec(word.lower())
            pretrained_embeddings[i] = torch.FloatTensor(vector)
        except:
            continue

    embeddings = nn.Embedding.from_pretrained(pretrained_embeddings)
    #deleting the loaded word2vec embedding
    del word_vectors
    logger.info("loaded the pretrained embeddings")

    return embeddings