# model.py
#  by: mika senghaas

from torch import nn
from torchvision.models import (
    alexnet,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)

from config import CLASSES, CNN_MODULES, RNN_MODULES

class BaseClassifier(nn.Module):
    """
    Abstract Class ImageClassifier.

    All classifier (ImageClassifier, VideoClassifier) classes should inherit
    from this class.

    Attributes:
        model_name (str): Identifier of the model to be used. Must be one of the keys
        num_classes (int): Number of classes in the dataset to finetune the model on.
        meta (dict): Dictionary containing relevant meta information about the model.

    Methods:
        check_preconditions (dict): Checks if the given kwargs are valid.
    """

    MANDATORY = ["model_name", "num_classes"]

    def __init__(self, kwargs):
        # init nn.Module class
        super().__init__()

        # check preconditions
        self.check_preconditions(kwargs)

        # parse mandatory kwargs
        self.model_name = kwargs["model_name"]
        self.num_classes = kwargs["num_classes"]

        # save kwargs into meta data
        self.meta = kwargs

    def check_preconditions(self, kwargs):
        # all mandatory kwargs are present
        assert len(kwargs.keys()) > 0 and all(
            x in kwargs.keys() for x in BaseClassifier.MANDATORY
        ), f"Class requires the parameters: {BaseClassifier.MANDATORY}"

        # check that model_name is valid
        self.check_model_name(kwargs["model_name"])

        # check that num_classes is valid
        self.check_num_classes(kwargs["num_classes"])

    def check_model_name(self, model_name: str) -> None:
        if "-" in model_name:
            cnn_module, rnn_module = model_name.split("-")
            assert cnn_module in CNN_MODULES
            assert rnn_module in RNN_MODULES
        else:
            assert model_name in CNN_MODULES

    def check_num_classes(self, num_classes: int) -> None:
        msg = f"num_classes must be > 0 and <= {len(CLASSES)}"
        assert num_classes > 0 and num_classes <= len(CLASSES), msg

    def get_cnn_module(self, cnn_module : str, num_classes: int) -> nn.Module:
        match cnn_module:
            case "resnet18":
                model = resnet18()
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            case "resnet50":
                model = resnet50()
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            case "alexnet":
                model = alexnet()
                model.classifier[6] = nn.Linear(
                    model.classifier[6].in_features, num_classes
                )
            case "mobilenet-v3-small":
                model = mobilenet_v3_small()
                model.classifier[3] = nn.Linear(
                    model.classifier[3].in_features, num_classes
                )
            case _:
                raise ValueError(f"CNN Module {self.model_name} not supported.")
        
        return model

    def get_rnn_module(self, 
           rnn_module : str, 
           input_size: int, 
           hidden_size: int, 
           num_layers: int) -> nn.Module:
        match rnn_module:
            case "rnn":
                model = nn.RNN(
                        input_size, 
                        hidden_size, 
                        num_layers=num_layers, 
                        batch_first=True)
            case "lstm":
                model = nn.LSTM(
                        input_size, 
                        hidden_size, 
                        num_layers=num_layers, 
                        batch_first=True)
            case _:
                raise ValueError(f"RNN Module {self.model_name} not supported.")
        
        return model


class ImageClassifier(BaseClassifier):
    """
    Class ImageClassifier.

    Attributes:
        model_name (str): Identifier of the model to be used. Must be one of the keys
                          of the MODELS/ WEIGHTS dict.
        num_classes (int): Number of classes in the dataset to finetune the model on.
        model (nn.Module): Finetuned model.
        num_params (int): Total number of parameters in the model.
        meta (dict): Dictionary containing relevant meta information about the model.

    Methods:
        default_config (dict): Returns a dict containing default model configuration.
        forward (torch.Tensor) -> torch.Tensor: Forward pass of the model.

    """

    @staticmethod
    def default_config():
        """
        Returns a dict containing the default configuration of the model.

        Returns:
            dict: Default configuration of the model.
        """
        model_name = "resnet18"
        num_classes = len(CLASSES)

        return {
            "model_name": model_name,
            "num_classes": num_classes,
        }

    def __init__(self, **kwargs):
        """
        Initialises a ImageClassifier instance.

        Takes a model identifier (as specified in keys of MODELS/ WEIGHTS dict) and
        number of classes as input and returns a nn.Module instance of the finetuned
        model. If the constructor is given a pretrained argument, the model will be
        initialised with the pretrained weights from the ImageNet dataset as
        downloadable from PyTorch. The model will be finetuned by replacing the last
        layer with a linear layer with the number of classes as output dimension.
        No weights are frozen, meaning that the finetuning will be done by training
        all weights of the model.

        Args:
            model_name (str): Identifier of the model to be used. Must be one of the
                              keys of the MODELS/ WEIGHTS dict.
            num_classes (int): Number of classes in the dataset to finetune the model.
            pretrained (bool, Optional): If True, the model will be initialised with
                                         the pretrained weights

        Returns:
            nn.Module: Finetuned model
        """
        super().__init__(kwargs)

        # initialise model
        self.model = self.get_cnn_module(self.model_name, self.num_classes)

        # compute total model parameters
        self.num_params = sum(param.numel() for param in self.model.parameters())

        # save relevant meta information
        self.meta.update({"num_params": self.num_params})

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.model(inputs)


class VideoClassifier(BaseClassifier):
    @staticmethod
    def default_config():
        """
        Returns a dict containing the default configuration of the model.

        Returns:
            dict: Default configuration of the model.
        """
        model_name = "resnet18/lstm"
        num_classes = len(CLASSES)

        return {
            "model_name": model_name,
            "num_classes": num_classes,
        }

    def __init__(self, **kwargs):
        """
        Initialises a VideoClassifier instance. Takes a model identifier of the form
        'cnn-module/rnn-module' and number of classes as input and returns a nn.Module.

        Args:
            model_name (str): Identifier of the model to be used.
            num_classes (int): Number of classes in the dataset to finetune the model.

        Returns:
            nn.Module: VideoClassifier instance
        """
        super().__init__(kwargs)

        # get name of cnn and rnn module
        cnn_module, rnn_module = self.model_name.split("-")

        # cnn module
        self.cnn_module = self.get_cnn_module(cnn_module, 256)

        # rnn module
        self.rnn_module = self.get_rnn_module(
                rnn_module, 
                input_size=256, 
                hidden_size=256, 
                num_layers=2)

        # classifier
        self.fc = nn.Linear(256, self.num_classes)

        # compute total model parameters
        self.num_params = sum(param.numel() for param in self.parameters())

        # update relevant meta information
        self.meta.update({"num_params": self.num_params})

    def forward(self, inputs):
        """
        Forward pass of the model to compute the logits given a batch of frame
        sequences.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, T, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, num_classes)
        """
        # pre conditions
        msg = "Input tensor must be (B, T, C, H, W), or (T, C, H, W)"
        assert inputs.ndim in [4, 5], msg
        transformed = False

        if inputs.ndim == 5:
            B, T, C, H, W = inputs.shape
            inputs = inputs.view(B * T, C, H, W)
            transformed = True

        # extract features from CNN
        cnn_features = self.cnn_module(inputs)
        if transformed:
            cnn_features = cnn_features.view(B, T, -1)

        # pass features through RNN
        out, _ = self.rnn_module(cnn_features)  # (B, T, H), (B, 1, H)

        return self.fc(out)
