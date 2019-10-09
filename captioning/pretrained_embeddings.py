import torch
import numpy as np
import torch.nn as nn
import logging
from gensim.models import KeyedVectors
from pathlib import Path

from config import Config

logger = logging.getLogger("captioning")

def load_pretrained_embeddings(pretrained_emb_path, id_to_word):
    """
    Loads the pretrained word2vec and initialzes the words from vocabulary using the embeddings.

    Args:
        pretrained_emb_path (Path): Path of the word2vec file.
        id_to_word (list): Vocabulary.

    Returns:
        Words of vocabulary initialized from pretrained word2vec embeddings.
    
    Raises:
        FileNotFoundError: If the pretrained word embeddings file doesn't exist.
    """
    # TODO (aq): Determine the dimensions of the embeddings on run time(not from the config file)
    logger.info("Loading the pretrained embeddings ...")
    emb_dim = Config.get("pretrained_emb_dim")
    #loading pretrained embeddings
    try:
        word_vectors = KeyedVectors.load_word2vec_format(pretrained_emb_path, binary=False)
    except FileNotFoundError:
        logger.exception("Traceback of pretrained embeddings file not found.")
    else:
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
        logger.info("Loaded the pretrained embeddings.")

        return embeddings
