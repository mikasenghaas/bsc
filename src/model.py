# model.py
#  by: mika senghaas

from torch import nn
from torchvision.models import (
    AlexNet_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    alexnet,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)

from config import *
from utils import *

MODELS = {
    'alexnet': alexnet,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'mobilenet-v3-small': mobilenet_v3_small
}

WEIGHTS = {
    'alexnet': AlexNet_Weights.DEFAULT,
    'resnet18': ResNet18_Weights.DEFAULT,
    'resnet50': ResNet50_Weights.DEFAULT,
    'mobilenet-v3-small': MobileNet_V3_Small_Weights.DEFAULT
}


class FinetunedImageClassifier(nn.Module):
    """
    Class FineTunedImageClassifier. 

    Attributes:
        model_name (str): Identifier of the model to be used. Must be one of the keys of the MODELS/ WEIGHTS dict.
        num_classes (int): Number of classes in the dataset to finetune the model on.
        model (nn.Module): Finetuned model.
        num_params (int): Total number of parameters in the model.
        meta (dict): Dictionary containing relevant meta information about the model.

    Methods:
        default_config (dict): Returns a dict containing the default configuration of the model.
        forward (torch.Tensor) -> torch.Tensor: Forward pass of the model.

    """
    MANDATORY = ['model_name', 'num_classes']

    @staticmethod
    def default_config():
        """
        Returns a dict containing the default configuration of the model.

        Returns:
            dict: Default configuration of the model.
        """
        model_name = "resnet18"
        pretrained = PRETRAINED
        num_classes = len(CLASSES)

        return {"model_name": model_name,
                "pretrained": pretrained,
                "num_classes": num_classes}

    def __init__(self, **kwargs):
        """
        Initialises a FinetunedImageClassifier instance.

        Takes a model identifier (as specified in keys of MODELS/ WEIGHTS dict) and number of classes as input and returns a nn.Module instance of the finetuned model. If the constructor is given a pretrained argument, the model will be initialised with the pretrained weights from the ImageNet dataset as downloadable from PyTorch. The model will be finetuned by replacing the last layer with a linear layer with the number of classes as output dimension. No weights are frozen, meaning that the finetuning will be done by training all weights of the model.

        Args:
            model_name (str): Identifier of the model to be used. Must be one of the keys of the MODELS/ WEIGHTS dict.
            num_classes (int): Number of classes in the dataset to finetune the model on.
            pretrained (bool, Optional): If True, the model will be initialised with the pretrained weights

        Returns:
            nn.Module: Finetuned model
        """
        super().__init__()
        # pre conditions
        assert len(kwargs.keys()) > 0 and all(x in kwargs.keys(
        ) for x in FinetunedImageClassifier.MANDATORY), f"Class requires the following parameters: {FinetunedImageClassifier.MANDATORY}"
        assert kwargs['model_name'] in MODELS, f"Choose model from {MODELS.keys()}"

        # parse mandatory kwargs
        self.model_name = kwargs["model_name"]
        self.num_classes = kwargs["num_classes"]

        # load pretrained weights if pretrained is set in kwargs
        if kwargs.get("pretrained"):
            weights = WEIGHTS[self.model_name]
        else:
            weights = None

        # initialise model
        self.model = MODELS[self.model_name](weights=weights)

        # replace last layer with linear layer with num_classes as output dimension
        if self.model_name in ['resnet18', 'resnet50']:  # resnet family
            self.model.fc = nn.Linear(
                self.model.fc.in_features,
                self.num_classes
            )

        elif self.model_name == 'alexnet':  # alexnet
            self.model.classifier[6] = nn.Linear(
                self.model.classifier[6].in_features,
                self.num_classes
            )

        elif self.model_name == 'mobilenet-v3-small':  # mobilenet
            self.model.classifier[3] = nn.Linear(
                self.model.classifier[3].in_features,
                self.num_classes
            )

        # compute total model parameters
        self.num_params = sum(param.numel()
                              for param in self.model.parameters())

        # save relevant meta information
        self.meta = kwargs
        self.meta.update({'num_params': self.num_params})

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.model(inputs)
