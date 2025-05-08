import torch
import torch.nn as nn
from .pointnet_utils import PointNetSetAbstraction, PointNetFeaturePropagation

def calc_acc(mos_pred, mos_gt):
    mos_pred_bin = (mos_pred > 0.5).float()
    acc = (mos_pred_bin == mos_gt).float().mean()
    return acc

def get_radar_velocity_vector(point, compensated_radial_velocity):
    distance = torch.norm(point)
    radial_unit_vectors = point / distance
    velocity_vector = compensated_radial_velocity * radial_unit_vectors
    return velocity_vector

def compensate_moving_points(ori_points, mos_pred, threshold=0.5):
    # ori_points: (N, 8) 第一列为batch_idx
    # 判定为运动的阈值？ 0.5
    count = 0
    updated_points = []
    for i, point in enumerate(ori_points):
        #  预测值大于0.5且为先前帧的就移动点
        if mos_pred[i] > threshold and point[7] != 0:
            count += 1
            batch_idx = point[0].view(1)
            xyz = point[1:4]
            frame_diff_sec = abs(point[7]) * 0.1
            radial_velocity = point[6]
            velocity_vector = get_radar_velocity_vector(xyz, radial_velocity)
            xyz = xyz + frame_diff_sec * velocity_vector
            # 将移动后点的t设为0，避免后续重复移动
            point[7] = 0
            updated_points.append(torch.cat((batch_idx, xyz, point[4:]), dim=0))
        else:
            updated_points.append(point)
    with open('/home/yu/OpenPCDet/move.txt', 'a') as f:
        f.write(f'move {count} points\n')
    updated_points = torch.stack(updated_points)
    return updated_points

"""
PointMOS
"""
# (N, 8) -> (B, N', 7)
def split_points_with_batch(all_points, all_labels, batch_size):
    batch_idx_unique = torch.unique(all_points[:, 0].int())

    batch_counts = [] # 记录每个 batch 有多少点
    batch_points_list = []
    batch_label_list = []

    for b_idx in batch_idx_unique:
        mask = (all_points[:, 0].int() == b_idx)
        points_b = all_points[mask]
        batch_counts.append(points_b.shape[0])
        batch_points_list.append(points_b[:, 1:])
        batch_label_list.append(all_labels[mask])

    M = max(batch_counts)
    # (B, M, 7)
    points_padded = torch.zeros((batch_size, M, 7), dtype=all_points.dtype, device=all_points.device)
    label_padded = torch.zeros((batch_size, M), dtype=all_labels.dtype, device=all_labels.device)
    for i, (pts7, count) in enumerate(zip(batch_points_list, batch_counts)):
        points_padded[i, :count, :] = pts7
        label_padded[i, :count] = batch_label_list[i]
    return points_padded, label_padded

def feature_encoding(point_cloud):
    # 分离坐标和特征
    coords = point_cloud[:, :3]  # x, y, z
    features = point_cloud[:, 3:]  # RCS, v_r, v_r_compensated, time
    # 强化v_r_compensated特征（第5个索引，从0开始计数）
    v_r_comp = features[:, 2:3]  # v_r_compensated
    # 创建额外的v_r_compensated派生特征
    v_r_comp_abs = torch.abs(v_r_comp) # 速度大小
    v_r_comp_squared = v_r_comp ** 2 # 强化速度本身
    v_r_comp_sign = torch.sign(v_r_comp) # 速度方向
    # 连接原始特征和增强的速度特征
    enhanced_features = torch.cat([
        coords,
        features,
        v_r_comp_abs,
        v_r_comp_squared,
        v_r_comp_sign
    ], dim=1)
    return enhanced_features # (N, 10)

class VelocityAttention(nn.Module):
    def __init__(self, in_channel):
        super(VelocityAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channel, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, in_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # v_idx是v_r_compensated在特征中的索引
        # x: B, 10, N
        v_feature = x[:, 5:6, :] # B, 1, N
        weights = self.mlp(x) # B, 10, N
        v_enhanced = weights[:, 5:6, :] * v_feature
        # B, 11, N
        x_enhanced = torch.cat([x, v_enhanced], dim=1)
        return x_enhanced  # B, 11, N

class PointMOS(nn.Module):
    def __init__(self, model_cfg):
        super(PointMOS, self).__init__()
        self.acc = 0
        self.model_cfg = model_cfg
        self.velocity_attention = VelocityAttention(in_channel=10)  # 11是增强后的特征数
        self.sa1 = PointNetSetAbstraction(
            npoint=1024,
            radius=0.2,
            nsample=32,
            in_channel=11 + 3,  # 10+1+3，因为速度注意力模块增加了一个特征，sample_and_group + 3
            mlp=[64, 64, 128],
            group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True
        )
        self.fp3 = PointNetFeaturePropagation(
            in_channel=1280,
            mlp=[256, 256]
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=384,
            mlp=[256, 128]
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + 11,
            mlp=[128, 128, 64]
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, batch_dict):
        # 去掉batch_idx (N, 7)
        points = batch_dict['points'][:,1:]
        # (N, 3) -> (1, N, 3)
        xyz = points[:, :3].unsqueeze(0)

        enhanced_features = feature_encoding(points).unsqueeze(0) # (B, N, 10)
        # (B, 10, N)
        enhanced_features_t = enhanced_features.transpose(1, 2)
        # (B, 11, N)
        attention_enhanced = self.velocity_attention(enhanced_features_t)
        # (B, N, 11)
        attention_enhanced = attention_enhanced.transpose(1, 2)
        l0_xyz = xyz
        l0_features = attention_enhanced

        # (B, n_point, 3) ->(B, n_point, C)
        l1_xyz, l1_features_sa = self.sa1(l0_xyz, l0_features) # [1, 1024, 128]
        l2_xyz, l2_features_sa = self.sa2(l1_xyz, l1_features_sa) # [1, 256, 256]
        l3_xyz, l3_features_sa = self.sa3(l2_xyz, l2_features_sa) # [1, 1, 1024]

        l2_features_fp = self.fp3(l2_xyz, l3_xyz, l2_features_sa, l3_features_sa) # [1, 256, 256]
        l1_features_fp = self.fp2(l1_xyz, l2_xyz, l1_features_sa, l2_features_fp) # [1, 1024, 128]
        l0_features_fp = self.fp1(l0_xyz, l1_xyz, attention_enhanced, l1_features_fp) # [1, N, 64] N为batch中radar点数

        x = self.classifier(l0_features_fp.transpose(1, 2))
        mos_pred = x.transpose(1, 2).squeeze() # (N,)

        self.acc = calc_acc(mos_pred, batch_dict['mos_label'])

        if self.acc >= 0.975:
            compensated_points = compensate_moving_points(batch_dict['points'], mos_pred, self.model_cfg.THRESHOLD)
        else:
            compensated_points = batch_dict['points']

        batch_dict.update({
            'mos_pred': mos_pred,
            'points': compensated_points,
            'mos_feature_sa': l1_features_sa,
            'mos_feature_fp': l1_features_fp
        })
        return batch_dict

    def get_loss(self, batch_dict):
        pred = batch_dict['mos_pred']
        target = batch_dict['mos_label']
        # num_pos = target.sum()
        # pos_weight = (target.numel() - num_pos) / (num_pos + 1e-6)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5))
        loss = criterion(pred, target)
        return loss


