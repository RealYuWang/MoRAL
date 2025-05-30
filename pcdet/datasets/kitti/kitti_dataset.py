import copy
import pickle
import numpy as np
from skimage import io
from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate
from pathlib import Path

class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode] # self.mode 默认 training
        # /home/yu/OpenPCDet/data/radar_5frames/training
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.root_split_path_lidar = Path('/home/yu/OpenPCDet/data/lidar/') / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)
        # lidar到radar坐标系的转换矩阵
        self.t_lidar_to_radar = np.array([[ 9.99940438e-01, 6.01582415e-03, 9.11667869e-03, -2.50410817e+00],
                                          [-6.03885078e-03, 9.99978696e-01, 2.49623171e-03, -4.26273974e-02],
                                          [-9.10150184e-03, -2.55114270e-03, 9.99955363e-01, 1.17628435e+00],
                                          [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_radar(self, idx):
        number_of_channels = 7
        radar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert radar_file.exists()
        points = np.fromfile(str(radar_file), dtype=np.float32).reshape(-1, number_of_channels)
        means = [0, 0, 0, -13.0, -3.0, -0.1, 0]
        stds = [1, 1, 1, 14.0, 8.0, 6.0, 1]
        points = (points - means) / stds
        return points

    def get_lidar(self, idx):
        lidar_file = self.root_split_path_lidar / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        lidar_points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        return lidar_points

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.jpg' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.jpg' % idx)
        assert img_file.exists(), f"{img_file} does not exist"
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_mos_label(self, idx):
        mos_label_file = self.root_split_path / 'label_mos' / ('%s.label' % idx)
        assert mos_label_file.exists(), f'{mos_label_file} not found'
        return np.fromfile(mos_label_file, dtype=np.uint8)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_calib_lidar(self, idx):
        calib_file = self.root_split_path_lidar / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        # 返回一个info字典，包含了该帧的点云和标注信息等
        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)
            calib_lidar = self.get_calib_lidar(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4, 'l2r': self.t_lidar_to_radar}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx) # 获取label中的所有objects
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                # 实际上是radar的？
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                loc_lidar_real = calib_lidar.rect_to_lidar(loc)
                l_l, h_l, w_l = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar_real[:, 2] += h_l[:, 0] / 2
                gt_boxes_lidar_real = np.concatenate([loc_lidar_real, l_l, w_l, h_l, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar_real'] = gt_boxes_lidar_real

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_radar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

                    ######

                    points_lidar = self.get_lidar(sample_idx)
                    calib_lidar = self.get_calib_lidar(sample_idx)
                    pts_rect_lidar = calib_lidar.lidar_to_rect(points_lidar[:, 0:3])

                    fov_flag_lidar = self.get_fov_flag(pts_rect_lidar, info['image']['image_shape'], calib_lidar)
                    pts_fov_lidar = points_lidar[fov_flag_lidar]
                    corners_lidar_real = box_utils.boxes_to_corners_3d(gt_boxes_lidar_real)
                    num_points_in_gt_lidar = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag_lidar = box_utils.in_hull(pts_fov_lidar[:, 0:3], corners_lidar_real[k])
                        num_points_in_gt_lidar[k] = flag_lidar.sum()
                    annotations['num_points_in_gt_lidar'] = num_points_in_gt_lidar

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        # 对sample_id_list里每一帧执行process_single_scene，返回字典列表
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # 读取infos里的每个info的信息，一个info是一帧的数据
        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_radar(sample_idx)
            points_lidar = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']
            # 有效object的个数
            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)
            point_indices_lidar = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points_lidar[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()

            for i in range(num_obj):
                filename_radar = '%s_%s_%s_%d.bin' % ('radar', sample_idx, names[i], i)
                filename_lidar = '%s_%s_%s_%d.bin' % ('lidar', sample_idx, names[i], i)
                filepath = database_save_path / filename_radar
                filepath_lidar = database_save_path / filename_lidar
                gt_points = points[point_indices[i] > 0]
                gt_points_lidar = points_lidar[point_indices_lidar[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                gt_points_lidar[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)
                with open(filepath_lidar, 'w') as f:
                    gt_points_lidar.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval
        from vod.evaluation import evaluate as vod_eval
        import os

        # print('------------------ Now kitti_eval.get_official_eval_result...')
        # kitti_eval的结果
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        # print('------------------ Now vod_eval_Evaluation.evaluate...')
        # vod_eval的结果
        eval_det_annos2 = copy.deepcopy(det_annos)
        eval_gt_annos2 = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        test_annotation_file = os.path.join("/datasets/vod/radar_5frames/training", 'label')
        vod_eval_Evaluation = vod_eval.Evaluation(test_annotation_file)
        results = vod_eval_Evaluation.evaluate(eval_gt_annos2, eval_det_annos2, class_names, current_class=[0, 1, 2])
        print("Results: \n"
              f"Entire annotated area: \n"
              f"Car: {results['entire_area']['Car_3d_all']} \n"
              f"Pedestrian: {results['entire_area']['Pedestrian_3d_all']} \n"
              f"Cyclist: {results['entire_area']['Cyclist_3d_all']} \n"
              f"mAP: {(results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3} \n"
              f"Driving corridor area: \n"
              f"Car: {results['roi']['Car_3d_all']} \n"
              f"Pedestrian: {results['roi']['Pedestrian_3d_all']} \n"
              f"Cyclist: {results['roi']['Cyclist_3d_all']} \n"
              f"mAP: {(results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3} \n"
              )

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']

        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        calib_lidar = self.get_calib_lidar(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_radar(sample_idx)
            mos_label = self.get_mos_label(sample_idx)

            if not self.training:
                input_dict['points_ori'] = points

            if self.dataset_cfg.FOV_POINTS_ONLY:
                # 转为校正后的相机坐标系
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                # 哪些点在相机FOV内
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                # print('Radar points before fov_flag: ', points.shape)
                points = points[fov_flag]
                mos_label = mos_label[fov_flag]
                # print('Radar points after fov_flag: ', points.shape)
            input_dict['points'] = points
            input_dict['mos_label'] = mos_label

            points_lidar = self.get_lidar(sample_idx)
            if not self.training:
                # predict时也要将lidar转换到radar坐标系，因为训练时就以radar坐标系为基准
                # 否则bbox会错位
                reflectivity = points_lidar[:, 3]
                points_loc = np.hstack((points_lidar[:, :3], np.ones((points_lidar.shape[0], 1), dtype=np.float32)))
                points_lidar_proj = np.dot(points_loc, self.t_lidar_to_radar.T)
                points_lidar_rcood = np.concatenate([points_lidar_proj, reflectivity[..., np.newaxis]], axis=1)
                input_dict['points_lidar_ori'] = points_lidar_rcood

            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect_lidar = calib_lidar.lidar_to_rect(points_lidar[:, 0:3])
                fov_flag_lidar = self.get_fov_flag(pts_rect_lidar, img_shape, calib_lidar)
                points_lidar = points_lidar[fov_flag_lidar]

            reflectivity = points_lidar[:, 3]
            points_loc = np.hstack((points_lidar[:, :3], np.ones((points_lidar.shape[0], 1), dtype=np.float32)))
            points_lidar_proj = np.dot(points_loc, self.t_lidar_to_radar.T)

            points_lidar_rcood = np.concatenate([points_lidar_proj, reflectivity[..., np.newaxis]], axis=1)

            input_dict['points_lidar'] = points_lidar_rcood


        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        # prepare_data 调用 data_processor, 处理一帧的data_dict
        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict

def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    # trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    # test_filename = save_path / 'kitti_infos_test.pkl'
    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    # with open(trainval_filename, 'wb') as f:
    #     pickle.dump(kitti_infos_train + kitti_infos_val, f)
    # print('Kitti info trainval file is saved to %s' % trainval_filename)

    # dataset.set_split('test')
    # kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(kitti_infos_test, f)
    # print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        # /home/yu/OpenPCDet
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'radar',
            save_path=ROOT_DIR / 'data' / 'radar'
        )
