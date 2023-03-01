# model.py
#  by: mika senghaas

from torch import nn
from torchvision.models import (
        alexnet, AlexNet_Weights,
        resnet18, ResNet18_Weights,
        resnet50, ResNet50_Weights,
        )


from config import *
from utils import *

MODELS = { 
          'alexnet': alexnet, 
          'resnet18': resnet18, 
          'resnet50': resnet50
        } 

WEIGHTS = { 
           'alexnet': AlexNet_Weights.DEFAULT, 
           'resnet18': ResNet18_Weights.DEFAULT, 
           'resnet50': ResNet50_Weights.DEFAULT
        } 

class BaseClassifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.meta = { k: v for k, v in kwargs.items()}
        if 'id2label' in self.meta:
            self.meta['num_classes'] = len(self.meta['id2label'])
            self.meta['label2id'] = { l: i for i, l in self.meta['id2label'].items()}

class FinetunedImageClassifier(BaseClassifier):
    def __init__(self, model_name : str, pretrained : bool, **kwargs):
        super().__init__(**kwargs)
        assert model_name in MODELS, f"Choose model from {MODELS.keys()}"

        # save parms
        self.model_name = model_name
        self.pretrained = pretrained

        model = MODELS[model_name]
        weights = WEIGHTS[model_name]

        # intialise model
        if self.pretrained: self.model = model(weights=weights)
        else: self.model = model()

        if self.model_name in ['resnet18', 'resnet50']:
            self.model.fc = nn.Linear(self.model.fc.in_features, self.meta['num_classes'])
        elif self.model_name == 'alexnet':
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, self.meta['num_classes'])

        # meta information to log
        self.meta = {
                'model_name': self.model_name,
                'pretrained': self.pretrained,
                'num_params': sum(param.numel() for param in self.model.parameters())
            }

    def forward(self, inputs):
        return self.model(inputs)
