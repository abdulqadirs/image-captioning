import logging

logger = logging.getLogger("captioning")

class Statistics:
    def __init__(self, outdir, tensorboard_writer):
        self.training_losses = []
        self.validation_losses = []
        self.testing_losses = []
        self.bleu_score = []
        self.tensorboard_writer = tensorboard_writer
        self.outdir = outdir
    
    def record(self, training_losses = None, validation_losses = None, testing_losses = None, bleu_score = None):
        """
        stores the statistics in lists
        """
        self.training_losses.append(training_losses) if training_losses is not None else {}
        self.validation_losses.append(validation_losses) if validation_losses is not None else {}
        self.testing_losses.append(testing_losses) if testing_losses is not None else {}
        self.bleu_score.append(bleu_score) if bleu_score is not None else {}
    
    def log_eval(self, epoch, dataset_name):
        """
        ouput the evaluation metrics
        """
        if dataset_name == "testing":
            logger.info("Average BLEU Score on Test Dataset")
            logger.info("[Eval: {}] {:6.2f} BLEU ".format(dataset_name, self.bleu_score[-1]))
        else:
            logger.info("[Eval: {}] Epoch: {} ".format(dataset_name, epoch))
            logger.info("[Eval: {}] {:6.2f} BLEU ".format(dataset_name, self.bleu_score[-1]))

    def log_losses(self, epoch):
        """
        outputs the loss of given epoch
        """
        logger.info("At epoch {}. Train Loss: {}".format(epoch, self.training_losses[-1]))
    
    def push_tensorboard_losses(self, epoch):
        """
        output losses to tensorboard
        """
        if self.training_losses:
            self.tensorboard_writer.add_scalar('losses/train', self.training_losses[-1], epoch)
        if self.validation_losses:
            self.tensorboard_writer.add_scalar('losses/validation', self.validation_losses[-1], epoch)
        if self.testing_losses:
            self.tensorboard_writer.add_scalar('losses/testing', self.testing_losses[-1], epoch)
    
    def push_tensorboard_eval(self, epoch, dataset_name):
        """
        output the bleu score to tensorboard
        """
        self.tensorboard_writer.add_scalar(dataset_name + "/BLEU", self.bleu_score[-1], epoch)
