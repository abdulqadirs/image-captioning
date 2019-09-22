import torch

def adam_optimizer(encoder, decoder, learning_rate):
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    return optimizer