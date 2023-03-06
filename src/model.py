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
    MANDATORY = ['model_name', 'num_classes'] 

    @staticmethod
    def default_config():
        model_name = "resnet18"
        pretrained = PRETRAINED
        num_classes = len(CLASSES)

        return {"model_name": model_name,
                "pretrained": pretrained,
                "num_classes": num_classes}

    def __init__(self, **kwargs):
        super().__init__()
        # pre conditions
        assert len(kwargs.keys()) > 0 and all(x in kwargs.keys() for x in FinetunedImageClassifier.MANDATORY), f"Class requires the following parameters: {FinetunedImageClassifier.MANDATORY}"
        assert kwargs['model_name'] in MODELS, f"Choose model from {MODELS.keys()}"

        # parse kwargs
        self.model_name = kwargs["model_name"]
        self.num_classes = kwargs["num_classes"]

        # initialise model
        if kwargs.get("pretrained"): 
            self.model = MODELS[self.model_name](weights=WEIGHTS[self.model_name])
        else: 
            self.model = MODELS[self.model_name]()

        if self.model_name in ['resnet18', 'resnet50']:
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        elif self.model_name == 'alexnet':
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, self.num_classes)

        # meta information
        self.meta = kwargs
        self.meta.update({
                'num_params': sum(param.numel() for param in self.model.parameters())
                })

    def forward(self, inputs):
        return self.model(inputs)
