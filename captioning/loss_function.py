import torch.nn as nn

def cross_entropy(predicted_captions, target_captions):
    """
    Calcualtes the cross entropy loss function using pytorch.
    
    Args:
        predicted_captions (tensor): captions predicted by model of shape(batch_size, captions_length, vocal_length).
        target_captions (tensor): reference captions of shape(batch_size, captions_lenth).
    
    Returns:
        Cross entory loss of shape(batch_size).
    """
    loss = nn.CrossEntropyLoss()
    batch_size, captions_length, vocab_length = predicted_captions.size()
    predicted_captions = predicted_captions.view(batch_size * captions_length, -1)
    target_captions = target_captions.view(-1)
    error = loss(predicted_captions, target_captions)

    return error

def l1_loss(predicted_captions, target_captions):
    """
    Calculate mean absolute error between each element in predicted and target captons.

    Args:
        predicted_captions (tensor):
        target_captions (tensor):
    
    Returns:
        l1_loss (float):
    """
    loss_function = nn.L1Loss()
    loss = loss_function(predicted_captions, target_captions)

    return loss

def mse_loss(predicted_captions, target_captions):
    """
    Measure the mean squared error between each element of predicted_captions and target_captions.

    Args:
        predicted_captions (tensor):
        target_captions (tensor):
    
    Returns:
        mse_loss (float):
    """
    loss_function = nn.MSELoss()
    loss = loss_function(predicted_captions, target_captions)

    return loss