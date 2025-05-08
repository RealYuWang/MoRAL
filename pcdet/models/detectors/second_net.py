from .detector3d_template import Detector3DTemplate


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # pred_dicts 为字典列表
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if batch_dict.get('mos_feature_sa', None) is not None:
                recall_dicts['mos_acc'] = self.mos.acc.item()
            return pred_dicts, recall_dicts

    # 最终训练损失
    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss() # self.dense_head: AnchorHeadSingle

        if batch_dict.get('mos_feature_sa', None) is not None:
            loss_mos = self.mos.get_loss(batch_dict)
            mos_acc = self.mos.acc
            tb_dict = {
                'loss_rpn': loss_rpn.item(),
                'loss_mos': loss_mos.item(),
                'mos_acc': mos_acc.item(),
                **tb_dict
            }
            loss = loss_rpn + loss_mos
        else:
            tb_dict = {
                'loss_rpn': loss_rpn.item(),
                **tb_dict
            }
            loss = loss_rpn

        return loss, tb_dict, disp_dict
