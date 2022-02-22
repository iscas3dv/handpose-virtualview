import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(dir)
sys.path.append(root)
from ops.render import depth_crop_expand
from ops.image_ops import normalize_depth_expand, normalize_depth
from ops.point_transform import transform_2D, transform_2D_to_3D, transform_3D
from models.a2j import A2J_model
from models.a2j_conf_net import A2JConfNet
import logging
logger = logging.getLogger(__file__)


class MultiviewA2J(nn.Module):
    def __init__(self, camera, num_joints, n_head, d_attn, d_k, d_v, d_inner, dropout_rate, num_select,
                 light=False, use_conf=True, random_select=False, random_sample=False):
        super(MultiviewA2J, self).__init__()
        self.camera = camera
        self.num_joints = num_joints
        self.n_head = n_head
        self.d_attn = d_attn
        self.d_k = d_k
        self.d_v = d_v
        self.d_inner = d_inner
        self.dropout_rate = dropout_rate
        self.num_select = num_select
        self.light = light
        self.use_conf = use_conf
        self.random_select = random_select
        self.random_sample = random_sample
        self.fx = camera["fx"]
        self.fy = camera["fy"]
        self.u0 = camera["u0"]
        self.v0 = camera["v0"]
        self.a2j = A2J_model(num_joints, dropout_rate=dropout_rate, light=light)
        self.conf_fuse_net = A2JConfNet(n_head, d_attn, d_k, d_v, d_inner, dropout_rate, num_select, random_select)

    def forward(self, cropped, crop_trans, com_2d, inter_matrix, cube, level, view_trans=None):
        """
        :param cropped: Tensor(B, 1, 176, 176) or Tensor(B, N, 1, 176, 176)
        :param crop_trans: Tensor(B, 3, 3)
        :param com_2d: Tensor(B, 3)
        :param inter_matrix: Tensor(B, 3, 3)
        :param cube: Tensor(B, 3)
        :param level: int
        :return:
            crop_expand: Tensor(B, num_views, 1, H, W)
            anchor_joints_2d_crop: Tensor(B, num_views, num_joints, 2)
            regression_joints_2d_crop: Tensor(B, num_views, num_joints, 2)
            depth_value_norm: Tensor(B, num_views, num_joints)
            joints_3d: Tensor(B, num_views, num_joints, 3)
            view_trans: Tensor(B, num_views, 4, 4)
            joint_3d_fused: Tensor(B, num_joints, 3)
            classification: Tensor(B*num_views, w/16*h/16*A, num_joints)
            regression: Tensor(B*num_views, w/16*h/16*A, num_joints, 2)
            depthregression: Tensor(B*num_views, w/16*h/16*A, num_joints)
        """
        if level==-1:
            assert view_trans is not None
            B, num_views, _, H, W = cropped.shape
            crop_expand = cropped
        else:
            B, _, H, W = cropped.shape
            if level>0:
                with torch.no_grad():
                    # crop_expand: Tensor(B, num_views, 1, H, W)
                    # view_trans: Tensor(B, num_views, 4, 4)
                    crop_expand, view_trans = depth_crop_expand(cropped, self.fx, self.fy, self.u0, self.v0, crop_trans,
                                                                level, com_2d, self.random_sample, False)
            elif level==0:
                if self.random_sample:
                    crop_expand, view_trans = depth_crop_expand(cropped, self.fx, self.fy, self.u0, self.v0, crop_trans,
                                                                level, com_2d, self.random_sample, False)
                else:
                    crop_expand = cropped[:, None, :, :, :]
                    view_trans = torch.eye(4, dtype=torch.float32)[None, None, :, :]
                    view_trans = view_trans.repeat((B, 1, 1, 1)).to(cropped.device)

            B, num_views, _, H, W = crop_expand.shape
            crop_expand = normalize_depth_expand(crop_expand, com_2d, cube)
        crop_expand = crop_expand.reshape((B * num_views, 1, H, W))


        # classification: (B*num_views, w/16*h/16*A, num_joints)
        # regression: (B*num_views, w/16*h/16*A, num_joints, 2)
        # depthregression: (B*num_views, w/16*h/16*A, num_joints)
        # anchor_joints_2d: (B*num_views, num_joints, 2)
        # regression_joints_2d: (B*num_views, num_joints, 2)
        # depth_value: (B*num_views, num_joints)
        classification, regression, depthregression, anchor_joints_2d_crop, regression_joints_2d_crop, \
        depth_value_norm = self.a2j(crop_expand)

        inv_corp_trans = torch.inverse(crop_trans)
        inv_corp_trans_expand = inv_corp_trans[:, None, :, :].repeat([1, num_views, 1, 1])
        inv_corp_trans_expand = inv_corp_trans_expand.reshape([-1, 3, 3])
        regression_joints_2d = transform_2D(regression_joints_2d_crop, inv_corp_trans_expand)
        com_z_expand = com_2d[:, 2][:, None].repeat([1, num_views]).reshape([B*num_views, 1])
        cube_z_expand = cube[:, 2][:, None].repeat([1, num_views]).reshape([B*num_views, 1])
        depth_value = depth_value_norm * cube_z_expand/2. + com_z_expand
        regression_joints_2d = regression_joints_2d.reshape([B, num_views, self.num_joints, 2])
        depth_value = depth_value.reshape([B, num_views, self.num_joints])
        # joints_3d_trans: (B, num_views, num_joints, 3)
        joints_3d_trans = torch.cat([regression_joints_2d, depth_value[..., None]], dim=-1)
        joints_3d_trans = transform_2D_to_3D(joints_3d_trans, self.fx, self.fy, self.u0, self.v0)
        joints_3d = transform_3D(joints_3d_trans, torch.inverse(view_trans))
        joint_3d_fused = torch.mean(joints_3d, dim=1)

        crop_expand = crop_expand.reshape((B, num_views, 1, H, W))
        anchor_joints_2d_crop = anchor_joints_2d_crop.reshape((B, num_views, self.num_joints, 2))
        regression_joints_2d_crop = regression_joints_2d_crop.reshape((B, num_views, self.num_joints, 2))
        depth_value_norm = depth_value_norm.reshape([B, num_views, self.num_joints])

        num_anchors = classification.shape[1]
        classification = torch.reshape(classification, (B, num_views, num_anchors, self.num_joints))
        regression = torch.reshape(regression, (B, num_views, num_anchors, self.num_joints, 2))
        depthregression = torch.reshape(depthregression, (B, num_views, num_anchors, self.num_joints))

        if self.use_conf:
            if level!=0:
                conf, joint_3d_conf = self.conf_fuse_net(classification, regression, depthregression, joints_3d)
            else:
                conf = torch.ones((B, 1), dtype=torch.float32)
                joint_3d_conf = joint_3d_fused
        else:
            joint_3d_conf = joint_3d_fused
            conf = None

        return crop_expand, anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm, joints_3d, view_trans,\
            joint_3d_fused, classification, regression, depthregression, conf, joint_3d_conf


if __name__ == '__main__':
    from feeders.nyu_feeder import NyuFeeder, collate_fn
    from torch.utils.data.dataloader import DataLoader
    import json

    dataset_config = json.load(open("../config/dataset/nyu.json", 'r'))
    train_dataset = NyuFeeder('train')
    dataloader = DataLoader(train_dataset, batch_size=6)
    predictor = MultiviewA2J(dataset_config["camera"], 14).cuda()
    for batch_idx, batch_data in enumerate(dataloader):
        item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = batch_data
        cropped = cropped.cuda()
        crop_trans = crop_trans.cuda()
        com_2d = com_2d.cuda()
        inter_matrix = inter_matrix.cuda()
        cube = cube.cuda()
        crop_expand, anchor_joints_2d, regression_joints_2d, depth_value, joints_3d, view_trans = \
            predictor(cropped, crop_trans, com_2d, inter_matrix, cube, level=4)
        print(crop_expand.shape)
        print(anchor_joints_2d.shape)
        print(regression_joints_2d.shape)
        print(depth_value.shape)
        print(joints_3d.shape)
        print(view_trans.shape)
        break
