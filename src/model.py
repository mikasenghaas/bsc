# model.py
#  by: mika senghaas

import torch
from torch import nn


class ImageClassifier(nn.Module):
    def __init__(self, hub_link, hub_identifier, pretrained, num_classes):
        super().__init__()

        model = torch.hub.load(hub_link, hub_identifier, pretrained=pretrained)

        match hub_identifier:
            case "alexnet":
                model.classifier[6] = nn.Linear(
                    model.classifier[6].in_features, num_classes
                )
            case "googlenet":
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            case "convnext_tiny":
                model.classifier[2] = nn.Linear(
                    model.classifier[2].in_features, num_classes
                )
            case "densenet121":
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            case "efficientnet_v2_s":
                model.classifier[1] = nn.Linear(
                    model.classifier[1].in_features, num_classes
                )
            case "resnet18":
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            case "resnet50":
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            case "mobilenet_v3_small":
                model.classifier[3] = nn.Linear(
                    model.classifier[3].in_features, num_classes
                )
            case "vit_b_16":
                model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            case _:
                raise ValueError(f"Model {hub_identifier} not supported.")

        self.model = model

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.model(inputs)


class VideoClassifier(nn.Module):
    def __init__(self, hub_link, hub_identifier, pretrained, num_classes):
        super().__init__()

        model = torch.hub.load(
            hub_link,
            hub_identifier,
            pretrained=pretrained,
        )

        match hub_identifier:
            case "r2plus1d_18":
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            case "slow_r50":
                fc = model.blocks[5].proj
                model.blocks[5].proj = nn.Linear(fc.in_features, num_classes)
            case "slowfast_r50":
                fc = model.blocks[6].proj
                model.blocks[6].proj = nn.Linear(fc.in_features, num_classes)
            case "x3d_s":
                fc = model.blocks[5].proj
                model.blocks[5].proj = nn.Linear(fc.in_features, num_classes)
            case _:
                raise NotImplementedError(
                    f"Model {hub_identifier} not implemented yet."
                )

        self.model = model

    def forward(self, inputs):
        """
        Forward pass of the model to compute the logits given a batch of frame
        sequences.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, T, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, num_classes)
        """
        return self.model(inputs)
