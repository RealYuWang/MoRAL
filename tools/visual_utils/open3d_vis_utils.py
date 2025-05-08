"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 0, 0], # 1 黑色 car
    [1, 0, 0], # 2 绿色 pedestrian [0, 0.5, 0]
    [1, 0.84, 0], # 3 金黄色 cyclist
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba

def points_to_spheres(points, radius=0.14):
    spheres = []
    for pt in points:
        mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_sphere.translate(pt)
        # 红色 [0.788, 0.216, 0.337]
        mesh_sphere.paint_uniform_color([0.141, 0.400, 0.286])
        spheres.append(mesh_sphere)
    return spheres

def draw_scenes(points, points_radar=None, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, draw_origin=True, frame_id=None, whichone=None):
    # 去掉B维度
    points = points.squeeze(0)
    if points_radar is not None:
        points_radar = points_radar.squeeze(0)
        # print(points_radar.shape)
    if gt_boxes is not None:
        gt_boxes = gt_boxes.squeeze(0)
        # 分离gt的box和label
        gt_labels = gt_boxes[:,-1].int()
        gt_boxes = gt_boxes[:, :7]

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(points_radar, torch.Tensor):
        points_radar = points_radar.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name=whichone, width=1936, height=1216)
    vis.get_render_option().point_size = 2.4
    vis.get_render_option().background_color = [1, 1, 1]

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.paint_uniform_color([0.188, 0.439, 0.702]) # LiDAR颜色
    vis.add_geometry(pts)

    if points_radar is not None:
        points_radar = points_radar[:, :3]
        spheres = points_to_spheres(points_radar)
        for s in spheres:
            vis.add_geometry(s)

    # 画BBox
    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1), gt_labels)
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    params = open3d.io.read_pinhole_camera_parameters(f'/home/yu/OpenPCDet/tools/camera_{frame_id}.json')
    view_ctl = vis.get_view_control()
    view_ctl.convert_from_pinhole_camera_parameters(params)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"/home/yu/OpenPCDet/predictions/{whichone}_predict_{frame_id}.png")
    vis.run()
    # view_ctl = vis.get_view_control()
    # open3d.io.write_pinhole_camera_parameters(f"/home/yu/OpenPCDet/tools/camera_{frame_id}.json", view_ctl.convert_to_pinhole_camera_parameters())
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    return line_set, box3d

def lineset_to_cylinders(line_set, radius=0.03, color=[0, 0, 1]):
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)
    cylinders = []

    for start_idx, end_idx in lines:
        start = points[start_idx]
        end = points[end_idx]
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue

        # 单位 cylinder 方向是 z 轴，从 (0,0,0) → (0,0,height)
        cylinder = open3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color(color)

        # Step 1: 对齐方向（z → direction）
        z_axis = np.array([0, 0, 1])
        axis = direction / length
        cross = np.cross(z_axis, axis)
        dot = np.dot(z_axis, axis)
        if np.linalg.norm(cross) < 1e-6:
            R = np.eye(3) if dot > 0 else -np.eye(3)  # 特例处理：反向
        else:
            skew = np.array([
                [0, -cross[2], cross[1]],
                [cross[2], 0, -cross[0]],
                [-cross[1], cross[0], 0]
            ])
            R = np.eye(3) + skew + skew @ skew * ((1 - dot) / (np.linalg.norm(cross) ** 2))

        # Step 2: 应用旋转
        cylinder.rotate(R, center=np.zeros(3))

        # Step 3: 平移到中点位置（不是起点！）
        midpoint = (start + end) / 2
        cylinder.translate(midpoint)

        cylinders.append(cylinder)

    return cylinders

def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        cylinders = lineset_to_cylinders(line_set, radius=0.05, color=[0, 1, 0])
        for cyl in cylinders:
            cyl.paint_uniform_color(box_colormap[ref_labels[i]])
            vis.add_geometry(cyl)
        # if ref_labels is None:
        #     line_set.paint_uniform_color(color)
        # else:
        #     line_set.paint_uniform_color(box_colormap[ref_labels[i]])
        # vis.add_geometry(line_set)
    return vis