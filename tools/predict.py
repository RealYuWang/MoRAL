import warnings
warnings.simplefilter("ignore")
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from visual_utils import open3d_vis_utils as V
import time


model_cfg_file = 'cfgs/kitti_models/RLNet.yaml'
cfg_from_yaml_file(model_cfg_file, cfg) # 自动加载了数据集配置文件到DATA_CONFIG下
logger = common_utils.create_logger()

test_set, test_loader, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=False, workers=8, logger=None, training=False
)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
model.load_params_from_file(filename='/home/yu/OpenPCDet/output/kitti_models/RLNet/default/ckpt/checkpoint_epoch_78.pth', logger=logger, to_cpu=True)
model.cuda()
model.eval()

# val_frame_ids = ['00000','00030','00365']
batch_dict = None
val_frame_id = '00000'
for batch in test_loader:
    if batch['frame_id'] == val_frame_id:
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
            frame_id=val_frame_id
        )
        # ours
        # V.draw_scenes(
        #     points=batch_dict['points_lidar_ori'][:, 1:],
        #     points_radar=batch_dict['points_ori'][:, 1:],
        #     ref_boxes=boxes,
        #     ref_scores=pred_dicts[0]['pred_scores'],
        #     ref_labels=labels,
        #     whichone='ours',
        #     frame_id=val_frame_id
        # )
else:
    print('frame not found!')