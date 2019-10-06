def detokenize_caption(tokenized_caption, id_to_word):
    """
    Converts the tokenized caption(consists of word ids from vocabulary) to english sentence.

    Args:
        tokenized_captions (list/tensor): captions consisting of word ids
        id_to_word (list): list of words in vocabulary indexed by word ids

    Returns:
        caption (list): caption of image in english (list of strings)
    """
    # TODO (aq): handling multiple captions
    # TODO (aq): <unk> token at the end of strings
    caption = []    
    for word_id in tokenized_caption:
        word = id_to_word[word_id]
        if word != '<pad>':
            caption.append(word)
        if word == '<end>':
            break

    return caption
