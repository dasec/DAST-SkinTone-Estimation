"""ViT-S/16 regressor predicting continuous CIELAB L*a*b* skin tone."""

import timm
import torch
import torch.nn as nn


def _get_feat_dim(backbone):
    """Feature width from a dummy forward pass; timm's num_features is unreliable."""
    from timm.data import resolve_data_config
    img_size = resolve_data_config(backbone.pretrained_cfg)["input_size"][1]
    backbone.eval()
    with torch.no_grad():
        out = backbone(torch.zeros(1, 3, img_size, img_size))
    return out.shape[-1]


class SkinToneRegressor(nn.Module):
    """Pretrained backbone plus a small MLP head; outputs [L*, a*, b*]."""

    def __init__(self, backbone_name="vit_small_patch16_224.augreg_in21k_ft_in1k",
                 freeze_backbone=True, num_outputs=3, pretrained=True):
        super().__init__()
        # pretrained=False skips the ImageNet download when a checkpoint is
        # loaded afterwards.
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0)
        feat_dim = _get_feat_dim(self.backbone)
        if pretrained:
            print(f"Backbone: {backbone_name}  feature_dim={feat_dim}  (ImageNet weights)")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.regression_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_outputs),
        )

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("Backbone unfrozen.")

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.regression_head(self.backbone(x))
