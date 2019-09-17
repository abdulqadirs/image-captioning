import torch
import torch.nn as nn

def criterion(predicted_captions, target_captions):
    loss = nn.CrossEntropyLoss()
    batch_size, captions_length, vocab_length = predicted_captions.size()
    predicted_captions = predicted_captions.view(batch_size * captions_length, -1)
    target_captions = target_captions.view(-1)
    error = loss(predicted_captions, target_captions)

    return error