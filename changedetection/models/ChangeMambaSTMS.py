import torch
import torch.nn as nn
import torch.nn.functional as F

from MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM
from MambaCD.classification.models.vmamba import LayerNorm2d
from MambaCD.changedetection.models.STMSDecoder import (
    MultiScaleSpatialEncoder,
    CrossTemporalStateInteraction,
    MultiScaleSpatioTemporalDecoder,
)


class ChangeMambaSTMS(nn.Module):
    def __init__(self, output_cd=2, pretrained=None, **kwargs):
        super().__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs["norm_layer"].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs["ssm_act_layer"].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs["mlp_act_layer"].lower(), None)

        clean_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["norm_layer", "ssm_act_layer", "mlp_act_layer"]
        }

        self.spatial_encoder = MultiScaleSpatialEncoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )
        self.temporal_interaction = CrossTemporalStateInteraction(
            num_scales=len(self.encoder.dims),
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )
        self.decoder = MultiScaleSpatioTemporalDecoder(
            num_scales=len(self.encoder.dims),
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )

        self.main_clf = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)

    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        pre_features = self.spatial_encoder(pre_features)
        post_features = self.spatial_encoder(post_features)

        fused_features, diff_features = self.temporal_interaction(pre_features, post_features)
        decoded = self.decoder(fused_features, diff_features)

        output = self.main_clf(decoded)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode="bilinear", align_corners=False)
        return output
