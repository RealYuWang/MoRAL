import warnings
warnings.simplefilter("ignore")
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from visual_utils import open3d_vis_utils as V
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Inference for OpenPCDet')
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--frame_id', type=str, default='00000', help='frame id in test set')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=8, logger=None, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    batch_dict = None
    for batch in test_loader:
        if batch['frame_id'] == args.frame_id:
            batch_dict = batch
            break
    
    if batch_dict is not None:
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            pred_dicts, _ = model.forward(batch_dict)
            boxes = pred_dicts[0]['pred_boxes']
            labels = pred_dicts[0]['pred_labels']
            # gt
            V.draw_scenes(
                points=batch_dict['points_lidar_ori'][:, 1:],
                points_radar=batch_dict['points_ori'][:, 1:],
                gt_boxes=batch_dict['gt_boxes'],
                whichone='gt',
                frame_id=args.frame_id
            )
            # ours
            V.draw_scenes(
                points=batch_dict['points_lidar_ori'][:, 1:],
                points_radar=batch_dict['points_ori'][:, 1:],
                ref_boxes=boxes,
                ref_scores=pred_dicts[0]['pred_scores'],
                ref_labels=labels,
                whichone='ours',
                frame_id=args.frame_id
            )
    else:
        print('frame not found!')