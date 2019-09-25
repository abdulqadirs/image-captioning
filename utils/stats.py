import logging

logger = logging.getLogger("captioning")

class Statistics:
    def __init__(self, outdir, tensorboard_writer):
        self.training_losses = []
        self.validation_losses = []
        self.testing_losses = []
        self.tensorboard_writer = tensorboard_writer
        self.outdir = outdir
    
    def record(self, training_losses = None, validation_losses = None, testing_losses = None):
        """
        stores the statistics in lists
        """
        self.training_losses.append(training_losses) if training_losses is not None else {}
        self.validation_losses.append(validation_losses) if validation_losses is not None else {}
        self.testing_losses.append(testing_losses) if testing_losses is not None else {}

    def get_losses(self, epoch):
        """
        outputs the loss of given epoch
        """
        logger.info("At epoch {}. Train Loss: {}".format(epoch, self.training_losses[-1]))
    
    def push_tensorboard_losses(self, epoch):
        """
        output losses to tensorboard
        """
        self.tensorboard_writer.add_scalar('losses/train', self.training_losses[-1], epoch)
        if self.validation_losses:
            self.tensorboard_writer.add_scalar('losses/validation', self.validation_losses[-1], epoch)
        if self.testing_losses:
            self.tensorboard_writer.add_scalar('losses/testing', self.testing_losses[-1], epoch)
