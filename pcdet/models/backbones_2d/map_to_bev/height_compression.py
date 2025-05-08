import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
import random

class SEAttention(nn.Module):
    def __init__(self, channel,ratio = 16):
        super(SEAttention, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
                nn.Linear(in_features=channel, out_features=channel // ratio),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=channel // ratio, out_features=channel),
                nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()   # 8, 256
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)

class MAGF(nn.Module):
    def __init__(self, lidar_channels=256, radar_feat_dim=128):
        super(MAGF, self).__init__()
        self.our_lambda = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.se = SEAttention(lidar_channels, ratio=16)
        self.fc_sa = nn.Sequential(
            nn.Linear(radar_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, lidar_channels)
        )
        self.fc_fp = nn.Sequential(
            nn.Linear(radar_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, lidar_channels)
        )
        self.gate_conv = nn.Conv2d(lidar_channels * 2, lidar_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, F_L, F_sa, F_fp):
        """
        Parameters:
          F_L  : [B, 256, 128, 128]
          F_sa : [1, 1024, 128] B,N,C
          F_fp : [1, 1024, 128]
        """
        B, C, H, W = F_L.shape
        F_L_se = self.se(F_L)  # [B, 256, 128, 128]

        F_sa_mean = F_sa.mean(dim=1)  # [1, 128]
        F_fp_mean = F_fp.mean(dim=1)  # [1, 128]
        F_sa_embed = self.fc_sa(F_sa_mean)  # [1, 256]
        F_fp_embed = self.fc_fp(F_fp_mean)  # [1, 256]
        radar_embed = self.our_lambda * F_sa_embed + (1 - self.our_lambda) * F_fp_embed
        # radar_embed = 0.5 * (F_sa_embed + F_fp_embed)  # [1, 256]
        radar_embed = radar_embed.expand(B, C)  # [B, 256]
        radar_embed_2d = radar_embed.view(B, C, 1, 1).expand(-1, -1, H, W)

        # concat [F_L_se, radar_embed_2d] -> 1x1 Conv -> Sigmoid
        fuse_input = torch.cat([F_L_se, radar_embed_2d], dim=1)  # [B, 2C, H, W]
        gate_map = self.sigmoid(self.gate_conv(fuse_input))
        F_fused = F_L_se + gate_map * F_L_se
        return F_fused

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        if self.model_cfg.WEIGHT:
            self.weight_conv = ConvModule(
                    2,
                    2,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01))
        self.magf = MAGF(lidar_channels=256, radar_feat_dim=128)

    def forward(self, batch_dict):
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        # -> BEV (B, 256, 128, 128)
        spatial_features = spatial_features.view(N, C * D, H, W)

        encoded_spconv_tensor_lidar = batch_dict['encoded_spconv_tensor_lidar']
        spatial_features_lidar = encoded_spconv_tensor_lidar.dense()
        N, C, D, H, W = spatial_features_lidar.shape
        # -> BEV (B, 256, 128, 128)
        spatial_features_lidar = spatial_features_lidar.view(N, C * D, H, W)
        mos_feature_sa = batch_dict['mos_feature_sa']
        mos_feature_fp = batch_dict['mos_feature_fp']
        # MAGF
        spatial_features_lidar = self.magf(spatial_features_lidar, mos_feature_sa, mos_feature_fp)

        if hasattr(self, 'weight_conv'):
            if self.training:
                rand_num = random.random()
                if rand_num > 0.8:
                    rand_drop = random.random()
                    if rand_drop > 0.8:
                        spatial_features_lidar = spatial_features_lidar * torch.zeros_like(spatial_features_lidar)
                    else:
                        spatial_features = spatial_features * torch.zeros_like(spatial_features)
            x_mean = torch.mean(spatial_features_lidar, dim=1, keepdim=True) # F_LA = AvgPool(F_L)
            r_x_mean = torch.mean(spatial_features, dim=1, keepdim=True) # F_RA = AvgPool(F_R)
            mix_x = torch.cat([x_mean, r_x_mean], dim=1) # F_mix = Concat[F_LA, F_RA]
            mix_x = self.weight_conv(mix_x)
            weight = torch.nn.functional.softmax(mix_x, dim=1) # [WL, WR] = Softmax(BN(Conv(F_mix))
            w1 = torch.split(weight, 1, dim=1)[0]
            w2 = torch.split(weight, 1, dim=1)[1]

            # (B, 512, 128, 128)
            fused_features = torch.cat([torch.mul(w1, spatial_features_lidar), torch.mul(w2, spatial_features)], dim=1)
            # if batch_dict.get('mos_feature', None) is not None:
            #     mos_feature = batch_dict['mos_feature'].view(1, 16, 128, 128)
            #     mos_feature = mos_feature.expand(batch_dict['batch_size'], 16, 128, 128)
            #     # (B, 528, 128, 128)
            #     fused_features = torch.cat([fused_features, mos_feature], dim=1)
            batch_dict['spatial_features'] = fused_features
            batch_dict['weight_radar'] = w1
        else:
            if batch_dict.get('encoded_spconv_tensor', None) is not None:
                batch_dict['spatial_features'] = spatial_features
            else:
                batch_dict['spatial_features'] = spatial_features_lidar
        return batch_dict