import torchvision
import torchvision.utils
import torch
import torch.nn as nn
import torchvision.models as vision_models

class Encoder(nn.Module):
    """
    Encodes the given image using CNNs(resnet18)

    Attributes:
        embed_size (int): The size of image embedding returned by CNN.
    """
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        #pretrained model
        self.resnet18 = vision_models.resnet18(pretrained=True)
        #replace the classifier with a fully connected embedding layer
        self.resnet18.classifier = nn.Linear(in_features=1000,  out_features=1000)
        #adding another fully connected layer
        self.embed = nn.Linear(in_features=1000, out_features=self.embed_size)
        #dropout layer
        self.dropout = nn.Dropout(p=0.5)
        #activation layer
        self.prelu = nn.PReLU()

    def forward(self, images):
        """
        Encodes the given images.

        Args:
            images (tensor): Images of shape (batch_size, image_shape)
        
        Returns:
            embeddings (tensor): Encoded images
        """
        #get the embeddings from resnet18
        resnet18_outputs = self.dropout(self.prelu(self.resnet18(images)))
        #passing through the fully connected layer
        embeddings = self.embed(resnet18_outputs)

        return embeddings
