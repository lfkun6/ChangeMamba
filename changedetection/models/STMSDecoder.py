import torch
import torch.nn as nn
import torch.nn.functional as F

from MambaCD.classification.models.vmamba import VSSBlock, Permute


class SpatialStateBlock(nn.Module):
    def __init__(self, in_channels, out_channels, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(
                hidden_dim=out_channels,
                drop_path=0.1,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=kwargs["ssm_d_state"],
                ssm_ratio=kwargs["ssm_ratio"],
                ssm_dt_rank=kwargs["ssm_dt_rank"],
                ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs["ssm_conv"],
                ssm_conv_bias=kwargs["ssm_conv_bias"],
                ssm_drop_rate=kwargs["ssm_drop_rate"],
                ssm_init=kwargs["ssm_init"],
                forward_type=kwargs["forward_type"],
                mlp_ratio=kwargs["mlp_ratio"],
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=kwargs["mlp_drop_rate"],
                gmlp=kwargs["gmlp"],
                use_checkpoint=kwargs["use_checkpoint"],
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleSpatialEncoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__()
        self.scale_blocks = nn.ModuleList(
            [
                SpatialStateBlock(
                    in_channels=dim,
                    out_channels=128,
                    channel_first=channel_first,
                    norm_layer=norm_layer,
                    ssm_act_layer=ssm_act_layer,
                    mlp_act_layer=mlp_act_layer,
                    **kwargs,
                )
                for dim in encoder_dims
            ]
        )
        self.aggregation_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
                for _ in range(len(encoder_dims) - 1)
            ]
        )

    def forward(self, features):
        encoded = [block(feature) for block, feature in zip(self.scale_blocks, features)]

        for idx in reversed(range(len(encoded) - 1)):
            upsampled = F.interpolate(
                encoded[idx + 1],
                size=encoded[idx].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            encoded[idx] = self.aggregation_blocks[idx](torch.cat([encoded[idx], upsampled], dim=1))

        return encoded


class TemporalStateBlock(nn.Module):
    def __init__(self, channels, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__()
        self.temporal_scan = nn.Sequential(
            nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(
                hidden_dim=channels,
                drop_path=0.1,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=kwargs["ssm_d_state"],
                ssm_ratio=kwargs["ssm_ratio"],
                ssm_dt_rank=kwargs["ssm_dt_rank"],
                ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs["ssm_conv"],
                ssm_conv_bias=kwargs["ssm_conv_bias"],
                ssm_drop_rate=kwargs["ssm_drop_rate"],
                ssm_init=kwargs["ssm_init"],
                forward_type=kwargs["forward_type"],
                mlp_ratio=kwargs["mlp_ratio"],
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=kwargs["mlp_drop_rate"],
                gmlp=kwargs["gmlp"],
                use_checkpoint=kwargs["use_checkpoint"],
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, pre_feat, post_feat):
        concat = torch.cat([pre_feat, post_feat], dim=1)
        temporal_context = self.temporal_scan(concat)
        gate = self.gate(concat)
        fused = gate * pre_feat + (1 - gate) * post_feat + temporal_context
        diff = torch.abs(pre_feat - post_feat)
        return fused, diff


class CrossTemporalStateInteraction(nn.Module):
    def __init__(self, num_scales, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__()
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalStateBlock(
                    channels=128,
                    channel_first=channel_first,
                    norm_layer=norm_layer,
                    ssm_act_layer=ssm_act_layer,
                    mlp_act_layer=mlp_act_layer,
                    **kwargs,
                )
                for _ in range(num_scales)
            ]
        )

    def forward(self, pre_features, post_features):
        fused_features = []
        diff_features = []
        for block, pre_feat, post_feat in zip(self.temporal_blocks, pre_features, post_features):
            fused, diff = block(pre_feat, post_feat)
            fused_features.append(fused)
            diff_features.append(diff)
        return fused_features, diff_features


class LocalTemporalRefinement(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, features, diff):
        attention = self.activation(self.pointwise(self.depthwise(diff)))
        return features + attention * diff


class MultiScaleSpatioTemporalDecoder(nn.Module):
    def __init__(self, num_scales, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__()
        self.merge_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_scales - 1)
            ]
        )
        self.decode_blocks = nn.ModuleList(
            [
                SpatialStateBlock(
                    in_channels=128,
                    out_channels=128,
                    channel_first=channel_first,
                    norm_layer=norm_layer,
                    ssm_act_layer=ssm_act_layer,
                    mlp_act_layer=mlp_act_layer,
                    **kwargs,
                )
                for _ in range(num_scales)
            ]
        )
        self.refine_blocks = nn.ModuleList([LocalTemporalRefinement(128) for _ in range(num_scales)])

    def forward(self, fused_features, diff_features):
        x = self.decode_blocks[-1](fused_features[-1])
        x = self.refine_blocks[-1](x, diff_features[-1])

        for idx in reversed(range(len(fused_features) - 1)):
            x = F.interpolate(x, size=fused_features[idx].shape[-2:], mode="bilinear", align_corners=False)
            x = self.merge_blocks[idx](torch.cat([x, fused_features[idx]], dim=1))
            x = self.decode_blocks[idx](x)
            x = self.refine_blocks[idx](x, diff_features[idx])

        return x
