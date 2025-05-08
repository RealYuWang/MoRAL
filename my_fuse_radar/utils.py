import os
import numpy as np
from vod.frame.transformations import homogeneous_transformation
from scipy.spatial import Delaunay
import open3d as o3d
#--------------------------
#        去除地面点
#--------------------------
def show_pcds_in_open3d(*args):
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud Visualization', width=1920, height=1080)
    opt = vis.get_render_option()
    opt.point_size = 1
    opt.background_color = np.asarray([0, 0, 0])
    for pcd in args:
        vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def remove_lidar_ground_points(bin_file):
    # load bin
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.05,
        ransac_n=3,
        num_iterations=1000,
    )
    [a, b, c, d] = plane_model
    print(f'Plane Equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0')
    first_ground_points = pcd.select_by_index(inliers)
    first_ground_points.paint_uniform_color([1, 0, 0]) # red
    non_ground_points = pcd.select_by_index(inliers, invert=True)
    # non_ground_points.paint_uniform_color([1, 1, 1]) # green

    plane_model2, inliers2 = non_ground_points.segment_plane(
        distance_threshold=0.05,
        ransac_n=3,
        num_iterations=1000,
    )
    second_ground_points = non_ground_points.select_by_index(inliers2)
    second_ground_points.paint_uniform_color([0, 0, 1])
    left_points = non_ground_points.select_by_index(inliers2, invert=True)
    left_points.paint_uniform_color([1, 1, 1])
    show_pcds_in_open3d(first_ground_points, second_ground_points, left_points)

#--------------------------
#        各类工具函数
#--------------------------
def is_point_in_box(point, obj, t_lidar_camera, t_camera_lidar):
    # camera 坐标系下
    bbox_vertices = get_transformed_bbox_corners(obj, t_lidar_camera, t_camera_lidar, dilate=True)
    box = Delaunay(bbox_vertices)
    if box.find_simplex(point) >= 0:
        return True
    else:
        return False

# 判断点是否在移动物体的bbox内
def if_points_in_moving_objects(point: np.ndarray, moving_objects, t_lidar_camera, t_camera_lidar):
    for m_obj in moving_objects:
        bbox_vertices = get_transformed_bbox_corners(m_obj, t_lidar_camera, t_camera_lidar)
        box = Delaunay(bbox_vertices)
        if box.find_simplex(point) >= 0:
            return True, m_obj['id']
    return False, -1

# 返回字典列表，每个字典包含一个object的信息 kitti有类似的函数
def read_kitti_label_file(raw_labels):
    static_objects = []
    moving_objects = []
    obj_id = 0
    for line in raw_labels:
        obj_id += 1
        obj = line.strip().split(' ')
        obj_data = {
            'id': obj_id,
            'type': obj[0],
            'movement': obj[1], # static or dynamic
            'height': float(obj[8]),
            'width': float(obj[9]),
            'length': float(obj[10]),
            'x': float(obj[11]), # camera coordinate
            'y': float(obj[12]),
            'z': float(obj[13]),
            'rotation': float(obj[14])
        }
        if obj[1] == '1' and obj[0] != 'DontCare':
            moving_objects.append(obj_data)
        else:
            static_objects.append(obj_data)
    return static_objects, moving_objects

def count_instances(json_path):
    count = 0
    with open(json_path, 'r') as f:
        for line in f:
            if line.strip():  # 忽略空行
                count += 1
    return count

def find_bad_frames(pose_dir):
    for json_file in os.listdir(pose_dir):
        json_path = os.path.join(pose_dir, json_file)
        if count_instances(json_path) != 3:
            print(f'Bad frame: {json_file}')
#--------------------------
#        画 3D BBox
#--------------------------
def get_base_box_vertices(m_obj):
    h, w, l= m_obj['height'], m_obj['width'], m_obj['length']
    x_offsets = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_offsets = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_offsets = [0, 0, 0, 0, h, h, h, h]
    return np.vstack([x_offsets, y_offsets, z_offsets])

def get_transformed_bbox_corners(m_obj, t_lidar_camera, t_camera_lidar, dilate=False):
    # rotation > 0 顺时针旋转
    x, y, z, rotation = m_obj['x'], m_obj['y'], m_obj['z'], -(m_obj['rotation'] + np.pi / 2)
    # LiDAR坐标系下bbox中心点，因为rotation是在LiDAR坐标系下的 绕z轴旋转
    bbox_center = (t_lidar_camera @ np.array([x, y, z, 1]))[:3]
    rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                           [np.sin(rotation), np.cos(rotation), 0],
                           [0, 0, 1]])
    # bbox的中心为原点，无旋转时的8个顶点坐标
    base_box_vertices = get_base_box_vertices(m_obj)
    new_corners_3d = np.dot(rot_matrix, base_box_vertices).T + bbox_center # (8, 3)

    if dilate: # 往外扩充一米
        direction = new_corners_3d - bbox_center
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        unit_dirs = direction / (norms + 1e-8)
        new_corners_3d = new_corners_3d + 1.5 * unit_dirs

    new_corners_3d_hom = np.concatenate((new_corners_3d, np.ones((8, 1))), axis=1)
    new_corners_3d_hom = homogeneous_transformation(new_corners_3d_hom, t_camera_lidar)[:,:3]#(8, 3) 转换回camera坐标系
    return new_corners_3d_hom

def plot_moving_objs_bbox(plot, moving_objs, t_lidar_camera, t_camera_lidar, color=0x006B88):
    import k3d
    for m_obj in moving_objs:
        transformed_bbox_corners = get_transformed_bbox_corners(m_obj, t_lidar_camera, t_camera_lidar)
        lines = [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [3, 7], [2, 6]]
        for plot_line in lines:
            plot += k3d.line(transformed_bbox_corners[plot_line], color=color, width=0.05)

def lidar_to_range_image(pc, height_resolution=256, width_resolution=1024):
    import matplotlib.pyplot as plt
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    distances = np.linalg.norm(pc, axis=1)
    elevation = np.arcsin(z / distances)
    azimuth = np.arctan2(y, x)

    elevation_min, elevation_max = -np.pi / 6, np.pi / 18
    azimuth_min, azimuth_max = -np.pi, np.pi

    range_image = np.zeros((height_resolution, width_resolution))
    # 映射点云到图像坐标
    elevation_scaled = ((elevation - elevation_min) / (elevation_max - elevation_min)) * (height_resolution - 1)
    azimuth_scaled = ((azimuth - azimuth_min) / (azimuth_max - azimuth_min)) * (width_resolution - 1)
    # 确保坐标在图像范围内
    elevation_indices = np.clip(elevation_scaled.astype(int), 0, height_resolution - 1)
    azimuth_indices = np.clip(azimuth_scaled.astype(int), 0, width_resolution - 1)
    range_image[elevation_indices, azimuth_indices] = distances

    plt.figure(figsize=(10, 5))
    plt.imshow(range_image, cmap='viridis', origin='upper')
    plt.colorbar(label='Range (m)')
    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    plt.title('Range Image')
    plt.show()


if __name__ == '__main__':
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="demo", width=1936, height=1216)
    vis.get_render_option().point_size = 2.5
    vis.get_render_option().background_color = [1,1,1]
    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # vis.add_geometry(axis_pcd)

    lidar_points = np.fromfile('/datasets/vod/lidar/training/velodyne/00000.bin', dtype=np.float32).reshape(-1, 4)[:,:3]
    radar_points = np.fromfile('/datasets/vod/radar_5frames/training/velodyne/00010.bin', dtype=np.float32).reshape(-1, 7)[:,:3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    pcd.paint_uniform_color([0.188, 0.439, 0.702])
    vis.add_geometry(pcd)

    params = o3d.io.read_pinhole_camera_parameters(f'/home/yu/OpenPCDet/tools/camera_00000.json')
    view_ctl = vis.get_view_control()
    view_ctl.convert_from_pinhole_camera_parameters(params)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("./lidar.png")


    vis.run()
    vis.destroy_window()

    # radar_points = np.fromfile('/datasets/vod/radar/training/velodyne/00100.bin', dtype=np.float32).reshape(-1,7)
    # print(radar_points[:,6])

