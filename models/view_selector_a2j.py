import torch
import torch.nn as nn
import os
import sys
dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(dir)
sys.path.append(root)
from models.multiview_a2j import MultiviewA2J
from ops.point_transform import transform_2D_to_3D
from ops.render import uniform_view_matrix, render_view, depth_crop_expand
from ops.image_ops import normalize_depth_expand


class ViewSelector(nn.Module):
    def __init__(self, multiview_a2j, conf_net, random):
        super().__init__()
        self.multiview_a2j = multiview_a2j
        self.conf_net = conf_net
        self.random = random

        self.multiview_a2j.eval()
        self.num_joints = self.multiview_a2j.num_joints
        self.camera = self.multiview_a2j.camera
        self.fx = self.camera["fx"]
        self.fy = self.camera["fy"]
        self.u0 = self.camera["u0"]
        self.v0 = self.camera["v0"]

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if module==self.conf_net:
                module.train(mode)
        return self

    def select(self, conf, k, random=False):
        """

        :param conf: Tensor(B, N)
        :param k: int
        :return:
            conf_select: Tensor(B, k)
            id_select: Tensor(B, k)
        """
        B, N = conf.shape
        if random:
            conf_select_list = []
            id_select_list = []
            for i in range(B):
                id = torch.arange(0, N, device=conf.device)
                id = id[torch.randperm(N)]
                id_select = id[:k]

                conf_select = conf[i, id_select]
                conf_select_list.append(conf_select)
                id_select_list.append(id_select)
            conf_select = torch.stack(conf_select_list, dim=0)
            id_select = torch.stack(id_select_list, dim=0)
        else:
            conf_select, id_select = torch.topk(conf, k, dim=-1) # (B, k)

        return conf_select, id_select

    def select_crop(self, crop_expand, view_trans, conf, k):
        """
        :param crop_expand: Tensor(B, N, 1, 176, 176)
        :param view_trans: Tensor(B, N, 4, 4)
        :param conf: Tensor(B, N)
        :param k: int
        :return:
            crop_select: Tensor(B, k, 1, 176, 176)
            joint_3d_select: Tensor(B, k, J, 3)
            conf_select: Tensor(B, k)
            id_select: Tensor(B, k)
        """
        B, N, _, W, H = crop_expand.shape
        conf_select, id_select = torch.topk(conf, k, dim=-1) # (B, k)

        id_select_expand = id_select[:, :, None, None, None].repeat((1, 1, 1, W, H))
        crop_select = torch.gather(crop_expand, 1, id_select_expand) # (B, k, 1, 176, 176)

        id_select_expand = id_select[:, :, None, None].repeat((1, 1, 4, 4))
        view_trans_select = torch.gather(view_trans, 1, id_select_expand)  # (B, k, 4, 4)

        return crop_select, view_trans_select, conf_select, id_select

    def forward(self, cropped, crop_trans, com_2d, inter_matrix, cube, level, k, inference):
        """
        :param cropped: Tensor(B, 1, 176, 176)
        :param crop_trans: Tensor(B, 3, 3)
        :param com_2d: Tensor(B, 3)
        :param inter_matrix: Tensor(B, 3, 3)
        :param cube: Tensor(B, 3)
        :param level: int
        :param k: int
        :inference: bool
        :return:
        """
        if level==1:
            self.shape = [1, 3]
        elif level==2:
            self.shape = [3, 3]
        elif level==3:
            self.shape = [3, 5]
        elif level==4:
            self.shape = [5, 5]
        elif level==5:
            self.shape = [9, 9]
        else:
            raise NotImplemented

        conf_light = self.conf_net(cropped)

        with torch.no_grad():
            conf_select_light, id_select_light = self.select(conf_light, k, self.random)
            crop_select_light, view_trans_select_light = depth_crop_expand(cropped, self.fx, self.fy, self.u0, self.v0,
                crop_trans, level, com_2d, False, random_ratote=False, indices=id_select_light)
            crop_select_light = normalize_depth_expand(crop_select_light, com_2d, cube)
            _, _, _, _, joints_3d_pred_select_light, _, joint_3d_fused_select_light, _, _, _, _, _ = \
                self.multiview_a2j(crop_select_light, crop_trans, com_2d, inter_matrix, cube, level=-1,
                                   view_trans=view_trans_select_light)
            conf_select_light = torch.softmax(conf_select_light, dim=-1)  # (B, k)

            joint_3d_conf_select_light = joints_3d_pred_select_light * conf_select_light[:, :, None, None]
            joint_3d_conf_select_light = torch.sum(joint_3d_conf_select_light, 1)
            if inference:
                return joints_3d_pred_select_light, joint_3d_fused_select_light, joint_3d_conf_select_light
            else:
                crop_expand, anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm, joints_3d_pred, \
                view_trans, joint_3d_fused, classification, regression, depthregression, conf, joint_3d_conf_select = \
                    self.multiview_a2j(cropped, crop_trans, com_2d, inter_matrix, cube, level=level)

                return crop_expand, view_trans, anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm, \
                       joints_3d_pred, joint_3d_fused, conf, joint_3d_conf_select, joints_3d_pred_select_light, \
                       joint_3d_fused_select_light, joint_3d_conf_select_light, conf_light
