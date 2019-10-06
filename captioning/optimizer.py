import torch

def adam_optimizer(encoder, decoder, learning_rate):
    """
    Returns the Adam Optimizer.

    Args:
        encoder (object): Image encoder (CNN).
        decoder (object): Image decoder (LSTM).
        learning_rate (float): Step size of optimizer.
    
    Returns:
        The Adam Optimizer.
    """
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    return optimizer