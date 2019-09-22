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


def load_checkpoint(checkpoint_file):
    """
    Loads the checkpoint of the model
    
    Params
    ------
    - checkpoint_file: file name of latest checkpoint file

    Return
    ------
    - checkpoint
    """
    if checkpoint_file.exists():
        print('Loading the checkpoint file')
        checkpoint = torch.load(str(checkpoint_file))
    else:
        print('No check point file exits')
    
    return checkpoint