import logging

logger = logging.getLogger("captioning")

class Statistics:
    """
    Recods the losses and evaluation metrics.
    Writes the losses and evaluation metrics to tensorboard.
    Logs the losses and evaluation metrics.

    Attributes:
        training_losses (list):
        validation_losses (list):
        testing_losses (list):
        bleu1 (list): evaluation metric
        bleu2 (list):
        bleu3 (list):
        bleu4 (list):
        tensorboard_writer (object):
        outdir (Path): output directory to store tensorboard log files.
    """
    def __init__(self, outdir, tensorboard_writer):
        self.training_losses = []
        self.validation_losses = []
        self.testing_losses = []
        self.bleu1 = []
        self.bleu2 = []
        self.bleu3 = []
        self.bleu4 = []
        self.tensorboard_writer = tensorboard_writer
        self.outdir = outdir
    
    def record(self, training_losses = None, validation_losses = None, testing_losses = None, 
                bleu1 = None,bleu2 = None, bleu3 = None, bleu4 = None):
        """
        Stores the statistics in lists.
        """
        self.training_losses.append(training_losses) if training_losses is not None else {}
        self.validation_losses.append(validation_losses) if validation_losses is not None else {}
        self.testing_losses.append(testing_losses) if testing_losses is not None else {}
        self.bleu1.append(bleu1 * 100) if bleu1 is not None else {}
        self.bleu2.append(bleu2 * 100) if bleu2 is not None else {}
        self.bleu3.append(bleu3 * 100) if bleu3 is not None else {}
        self.bleu4.append(bleu4 * 100) if bleu4 is not None else {}

    
    def log_eval(self, epoch, dataset_name):
        """
        Ouput the evaluation metrics to stout and logfile.
        """
        if dataset_name == "testing":
            logger.info("[Eval: {}] {:6.2f} BLEU-1 (%)".format(dataset_name, self.bleu1[-1]))
            logger.info("[Eval: {}] {:6.2f} BLEU-2 (%)".format(dataset_name, self.bleu2[-1]))
            logger.info("[Eval: {}] {:6.2f} BLEU-3 (%)".format(dataset_name, self.bleu3[-1]))
            logger.info("[Eval: {}] {:6.2f} BLEU-4 (%)".format(dataset_name, self.bleu4[-1]))

        else:
            logger.info("[Eval: {}] Epoch: {} ".format(dataset_name, epoch))
            logger.info("[Eval: {}] {:6.2f} BLEU-1 (%)".format(dataset_name, self.bleu1[-1]))
            logger.info("[Eval: {}] {:6.2f} BLEU-2 (%)".format(dataset_name, self.bleu2[-1]))
            logger.info("[Eval: {}] {:6.2f} BLEU-3 (%)".format(dataset_name, self.bleu3[-1]))
            logger.info("[Eval: {}] {:6.2f} BLEU-4 (%)".format(dataset_name, self.bleu4[-1]))

    def log_losses(self, epoch):
        """
        Outputs the loss of given epoch to stout and logfile.
        """
        logger.info("At epoch {}. Train Loss: {}".format(epoch, self.training_losses[-1]))
    
    def push_tensorboard_losses(self, epoch):
        """
        Output losses to tensorboard.
        """
        if self.training_losses:
            self.tensorboard_writer.add_scalar('losses/train', self.training_losses[-1], epoch)
        if self.validation_losses:
            self.tensorboard_writer.add_scalar('losses/validation', self.validation_losses[-1], epoch)
        if self.testing_losses:
            self.tensorboard_writer.add_scalar('losses/testing', self.testing_losses[-1], epoch)
    
    def push_tensorboard_eval(self, epoch, dataset_name):
        """
        Output the bleu score to tensorboard.
        """
        self.tensorboard_writer.add_scalar(dataset_name + "/BLEU-1", self.bleu1[-1], epoch)
        self.tensorboard_writer.add_scalar(dataset_name + "/BLEU-2", self.bleu2[-1], epoch)
        self.tensorboard_writer.add_scalar(dataset_name + "/BLEU-3", self.bleu3[-1], epoch)
        self.tensorboard_writer.add_scalar(dataset_name + "/BLEU-4", self.bleu4[-1], epoch)
