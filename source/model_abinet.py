import logging
import torch.nn as nn
from fastai.vision import *

from modules.attention import *
from modules.backbone import ResTranformer
from modules.model import Model
from modules.resnet import resnet45


class BaseVision_DomainDiscriminator(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = ifnone(config.model_vision_loss_weight, 1.0)
        self.out_channels = ifnone(config.model_vision_d_model, 512)

        if config.model_vision_backbone == 'transformer':
            self.backbone = ResTranformer(config)
        else: self.backbone = resnet45()

        """ Binary classifier"""
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1)
        )
        self.AdaptiveAvgPool_2 = nn.AdaptiveAvgPool2d((None, 1))
        self.predict = nn.Linear(self.out_channels, 1)

        if config.model_vision_checkpoint is not None:
            logging.info(f'Read vision model from {config.model_vision_checkpoint}.')
            self.load(config.model_vision_checkpoint, strict = False)

    def forward(self, images, *args):
        features = self.backbone(images)  # (N, E, H, W)
        features = features.permute(0, 3, 1, 2)  # (N, E, H, W) -> (N, W, E, H)
        features = self.AdaptiveAvgPool(features)  # (N, W, E, 1)
        features = features.squeeze(3)  # (N, W, E)

        features = features.permute(0, 2, 1)  # (N, W, E) -> (N, E, W)
        features = self.AdaptiveAvgPool_2(features)  # (N, E, 1)
        features = features.squeeze(2)  # (N, E)

        logits = self.predict(features)  # (N, 1)

        return logits
