import torch

def greedy_search(predicted_captions):
    """
    Converts the predicted caption of shape
    (batch_size, captions_length, vocab_length) to (batch_size, captions_length)
    by selecting the words of vocabulary with max(probability)

    Args:
        predicted_captions (tensor): captions predicted by model of shape(batch_size, captions_length, vocab_length)

    Returns:
        tokenized_caption (list): tokenized captions of shape (batch_size, captions_length)
    """
    batch_size, captions_length, vocab_length = predicted_captions.size()
    tokenized_caption = []
    for i in range(batch_size):
        for j in range(captions_length):
            values, indices = torch.max(predicted_captions[i][j], 0)
            tokenized_caption.append(indices.item())
    
    return tokenized_caption