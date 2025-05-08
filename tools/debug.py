import warnings
warnings.simplefilter("ignore")
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


model_cfg_file = 'cfgs/kitti_models/RLNet.yaml'
cfg_from_yaml_file(model_cfg_file, cfg) # 自动加载了数据集配置文件到DATA_CONFIG下
logger = common_utils.create_logger()


# 占用大部分加载时间
train_set, train_loader, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=8,
    dist=False, workers=8,
    logger=None,
    training=True
)

batch = next(iter(train_loader))
load_data_to_gpu(batch)
print(batch['points'].size())

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
model.cuda()
model.train()

ret_dict, tb_dict, disp_dict = model(batch)
print(ret_dict)
print(tb_dict)




