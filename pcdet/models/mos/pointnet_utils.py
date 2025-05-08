import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 分批次计算点与点之间的欧几里得距离，返回距离矩阵
def square_distance(src, dst):
    # dim=-1 对最后一个维度
    # src (B, N, 1, C)   dst (B, 1, M, C) 扩展维度方便广播操作
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

# 根据索引提取点云中的点
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
        如果 idx 形状为 [B, S]，采样 S 个点的索引
        如果 idx 形状为 [B, S, K]，表示从每组中采样 K 个点，一共 S*K个点
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    # print(f'points.shape: {points.shape}, idx.shape: {idx.shape}')


    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1) # 转为二维
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

# 最远点采样, 返回采样到的点的索引
def farthest_point_sample(xyz, n_point):
    """
    Input:
        xyz: point cloud data, [B, N, 3]
        n_point: number of samples
    Return:
        centroids: sampled point cloud index, [B, n_point]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, n_point, dtype=torch.long).to(device)

    distance = torch.ones(B, N).to(device) * 1e10 # 每个点到已采样点的最小距离，初始化为大值
    # 0 ~ N 之间随机选三个
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # 初始化一个最远点的
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(n_point):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 更新最短距离列表 之前求的已经存储下来，只需要和最新的距离比较取小值就好
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

# PointNet++ 局部区域点的查询
def query_ball_point(radius, n_sample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        n_sample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, n_sample]
        返回每个查询点对应的邻居的索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    # (N,) -> (1, 1, N) -> (B, S, N)
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1]) # 初始索引
    sqr_dists = square_distance(new_xyz, xyz) # 所有点到查询点的距离
    group_idx[sqr_dists > radius ** 2] = N # 标记超过距离约束的点
    group_idx = group_idx.sort(dim=-1)[0][:, :, :n_sample] # 按距离排序并截取
    # 如果不足n sample，用最近点填充
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, n_sample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(n_point, radius, n_sample, xyz, points, return_fps=False, knn=False):
    """
    Input:
        n_point: 采样点数量 S
        radius: 局部区域半径，用于 ball query
        n_sample: 每个ball内最多选择几个点
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D] D为特征维度
    Return:
        new_xyz: sampled points position data, [B, n_point, n_sample, 3]
        new_points: sampled points data, [B, n_point, n_sample, 3+D]
    """
    B, N, C = xyz.shape
    S = n_point
    fps_idx = farthest_point_sample(xyz, n_point) # [B, n_point]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x n_point x N
        idx = dists.argsort()[:, :, :n_sample]  # B x n_point x K
    else:
        idx = query_ball_point(radius, n_sample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, n_point, n_sample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, n_point, n_sample, C+D]
    else:
        new_points = grouped_xyz_norm
    if return_fps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

# 下采样，用于 Transition Down
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, knn=self.knn)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        return new_xyz, new_points

# PointNet++ 的上采样层，用于 Transition Up
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    # 从采样层xyz2中获取低分辨率特征point2，插值回原始点云xyz1的高分辨率特征空间
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: 原始点云位置, [B, N, C]
            xyz2: 采样后点云位置, [B, S, C]
            points1: 原始点云特征, [B, N, D]
            points2: 采样后点云的特征, [B, S, D]
        Return:
            new_points: 上采样后的特征, [B, D', N]
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape # S是采样后点的个数

        # xyz2.shape[1] 应该等于 points2.shape[1]
        # print(S)
        # print(points2.shape[1])

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1) #
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            # xyz1中每个点在xyz2中最近的3个点的索引
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            # points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)
