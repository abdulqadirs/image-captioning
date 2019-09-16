import configparser
from data import read_captions, dictionary


def decode_captions(tokenized_caption):
    config = configparser.ConfigParser()
    config.read('config.ini')
    captions_path = config['paths']['captions_dir']
    raw_captions = read_captions(captions_path)
    id_to_word, word_to_id = dictionary(raw_captions, threshold = 5)
    caption = []
    for word_id in tokenized_caption:
        word = id_to_word[word_id]
        if word != '<pad>':
            caption.append(word)
        if word == '<end>':
            break
    return caption



