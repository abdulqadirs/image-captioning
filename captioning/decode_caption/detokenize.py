def detokenize_caption(tokenized_caption, id_to_word):
    """
    Converts the tokenized caption(consists of word ids from vocabulary) to english sentence

    Params
    ------
    - tokenized_captions: captions consisting of word ids
    - id_to_word: list of words in vocabulary indexed by word ids

    Returns
    -------
    - caption: caption of image in english 
    """
    #needs to be fixed for multiple captions
    caption = []    
    for word_id in tokenized_caption:
        word = id_to_word[word_id]
        if word != '<pad>':
            caption.append(word)
        if word == '<end>':
            break

    return caption
