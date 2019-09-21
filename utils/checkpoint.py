import torch

def save_checkpoint(epoch, outdir, encoder, decoder, optimizer, criterion, 
                    filename = 'checkpoint.ImageCaptioning.pth.tar'):
    
    """
    saves the encoder and decoder checkpoint

    Params
    ------
    - epoch: current epoch
    - outdir: directory to output checkpoint
    - encoder: the encoder to save
    - decoder: the decoder to save
    - optimizer: optimizer for the model
    - filename: for checkpoint in outdir

    Return
    ------
    """
    filename = outdir +  '/' +  filename
    torch.save({'epoch': epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, str(filename))
    