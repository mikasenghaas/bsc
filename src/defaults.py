DEFAULT = {
    "alexnet": {
        "general": {
            "name": "AlexNet",
            "desc": "Classical Deep CNN for Image Classification",
            "type": "image",
            "link": "https://pytorch.org/vision/stable/models.html"
            "#torchvision.models.alexnet",
        },
        "model": {
            "hub_link": "pytorch/vision",
            "hub_identifier": "alexnet",
            "pretrained": False,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_size": 224,
        },
        "dataset": {},
        "loader": {
            "batch_size": 32,
            "shuffle": True,
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "trainer": {"epochs": 10, "device": "mps"},
    },
    "googlenet": {
        "general": {
            "name": "Google LeNet",
            "desc": "TBD",
            "type": "image",
            "link": "TBD"
        },
        "model": {
            "hub_link": "pytorch/vision",
            "hub_identifier": "googlenet",
            "pretrained": False,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_size": 256,
        },
        "dataset": {},
        "loader": {
            "batch_size": 32,
            "shuffle": True,
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "trainer": {"epochs": 10, "device": "mps"},
    },
    "convnext_tiny": {
        "general": {
            "name": "ConvNext Tiny",
            "desc": "A CNN for Image Classification for the 2020s",
            "type": "image",
            "link": "https://pytorch.org/vision/main/models/generated/torchvision."\
                    "models.convnext_tiny.html#torchvision.models.ConvNeXt_Tiny_Weights"
        },
        "model": {
            "hub_link": "pytorch/vision",
            "hub_identifier": "convnext_tiny",
            "pretrained": False,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_size": 224,
        },
        "dataset": {},
        "loader": {
            "batch_size": 32,
            "shuffle": True,
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "trainer": {"epochs": 10, "device": "mps"},
    },
    "densenet121": {
        "general": {
            "name": "DenseNet121",
            "desc": "TBD",
            "type": "image",
            "link": "https://pytorch.org/vision/main/models/generated/torchvision."\
                    "models.densenet121.html#torchvision.models.DenseNet121_Weights"
        },
        "model": {
            "hub_link": "pytorch/vision",
            "hub_identifier": "densenet121",
            "pretrained": False,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_size": 224,
        },
        "dataset": {},
        "loader": {
            "batch_size": 32,
            "shuffle": True,
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "trainer": {"epochs": 10, "device": "mps"},
    },
    "efficientnet_v2_s": {
        "general": {
            "name": "EfficientNet V2 Small",
            "desc": "TBD",
            "type": "image",
            "link": "TBD"
        },
        "model": {
            "hub_link": "pytorch/vision",
            "hub_identifier": "efficientnet_v2_s",
            "pretrained": False,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_size": 224,
        },
        "dataset": {},
        "loader": {
            "batch_size": 32,
            "shuffle": True,
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "trainer": {"epochs": 10, "device": "mps"},
    },
    "mobilenet_v3_small": {
        "general": {
            "name": "MobileNet V3 Small",
            "desc": "TBD",
            "type": "image",
            "link": "TBD"
        },
        "model": {
            "hub_link": "pytorch/vision",
            "hub_identifier": "mobilenet_v3_small",
            "pretrained": False,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_size": 224,
        },
        "dataset": {},
        "loader": {
            "batch_size": 32,
            "shuffle": True,
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "trainer": {"epochs": 10, "device": "cpu"},
    },
    "resnet18": {
        "general": {
            "name": "ResNet18",
            "desc": "CNN with Residual Connections for Image Classification",
            "type": "image",
            "link": "https://pytorch.org/vision/main/models/generated/"
            "torchvision.models.resnet18.html#torchvision.models.resnet18",
        },
        "model": {
            "hub_link": "pytorch/vision",
            "hub_identifier": "resnet18",
            "pretrained": False,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_size": 224,
        },
        "dataset": {},
        "loader": {
            "batch_size": 32,
            "shuffle": True,
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "trainer": {"epochs": 10, "device": "mps"},
    },
    "resnet50": {
        "general": {
            "name": "ResNet 50",
            "desc": "TBD",
            "type": "image",
            "link": "TBD"
        },
        "model": {
            "hub_link": "pytorch/vision",
            "hub_identifier": "resnet50",
            "pretrained": False,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_size": 224,
        },
        "dataset": {},
        "loader": {
            "batch_size": 32,
            "shuffle": True,
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "trainer": {"epochs": 10, "device": "mps"},
    },
    "vit_b_16": {
        "general": {
            "name": "Vision Transformer Base-16",
            "desc": "Vision Transformer (ViT) model with 16x16 patch resolution",
            "type": "image",
            "link": "https://pytorch.org/vision/main/models/generated/torchvision."\
                    "models.vit_b_16.html#torchvision.models.ViT_B_16_Weights"
        },
        "model": {
            "hub_link": "pytorch/vision",
            "hub_identifier": "vit_b_16",
            "pretrained": False,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_size": 224,
        },
        "dataset": {},
        "loader": {
            "batch_size": 32,
            "shuffle": True,
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "trainer": {"epochs": 10, "device": "mps"},
    },
    "x3d_s": {
        "general": {
            "name": "X3D",
            "desc": "3D CNN for Video Classification",
            "type": "video",
            "link": "https://pytorchvideo.readthedocs.io/en/latest/api/models/x3d.html",
        },
        "model": {
            "hub_link": "facebookresearch/pytorchvideo",
            "hub_identifier": "x3d_s",
            "pretrained": True,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.45, 0.45, 0.45],
            "std": [0.225, 0.225, 0.225],
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 6,
        },
        "dataset": {
            "clip_duration": 13 * 6 / 30,
        },
        "loader": {
            "batch_size": 4,
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "trainer": {"epochs": 10, "device": "cpu"},
    },
    "r2plus1d_18": {
        "general": {
            "name": "R(2+1)D",
            "desc": "ResNet based Video Classification",
            "type": "video",
            "link": "https://pytorchvideo.readthedocs.io/en/latest/api/models/x3d.html",
        },
        "model": {
            "hub_link": "pytorch/vision",
            "hub_identifier": "r2plus1d_18",
            "pretrained": True,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.45, 0.45, 0.45],
            "std": [0.225, 0.225, 0.225],
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 6,
        },
        "dataset": {
            "clip_duration": 13 * 6 / 30,
        },
        "loader": {
            "batch_size": 4,
        },
        "optim": {
            "lr": 0.0001,
            "weight_decay": 1e-4,
        },
        "trainer": {"epochs": 10, "device": "cpu"},
    },
    "slow_r50": {
        "general": {
            "name": "SlowFast R50",
            "desc": "Slow R50 for video classification",
            "type": "video",
            "link": "https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/",
        },
        "model": {
            "hub_link": "facebookresearch/pytorchvideo",
            "hub_identifier": "slow_r50",
            "pretrained": True,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.45, 0.45, 0.45],
            "std": [0.225, 0.225, 0.225],
            "side_size": 182,
            "crop_size": 256,
            "num_frames": 8,
            "sampling_rate": 8,
        },
        "dataset": {
            "clip_duration": 8 * 8 / 30,
        },
        "loader": {
            "batch_size": 4,
        },
        "optim": {
            "lr": 0.0001,
            "weight_decay": 1e-4,
        },
        "trainer": {"epochs": 10, "device": "cpu"},
    },
    "slowfast_r50": {
        "general": {
            "name": "SlowFast R50",
            "desc": "SlowFast R50 for video classification",
            "type": "video",
            "link": "https://pytorchvideo.readthedocs.io/en/"\
                    "latest/api/models/slowfast.html",
        },
        "model": {
            "hub_link": "facebookresearch/pytorchvideo",
            "hub_identifier": "slowfast_r50",
            "pretrained": True,
            "num_classes": 20,
        },
        "transform": {
            "mean": [0.45, 0.45, 0.45],
            "std": [0.225, 0.225, 0.225],
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 8,
            "sampling_rate": 8,
        },
        "dataset": {
            "clip_duration": 8 * 8 / 30,
        },
        "loader": {
            "batch_size": 4,
        },
        "optim": {
            "lr": 0.0001,
            "weight_decay": 1e-5,
        },
        "trainer": {"epochs": 10, "device": "cpu"}
    },
}
