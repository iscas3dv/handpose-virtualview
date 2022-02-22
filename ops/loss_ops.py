import torch
import torch.nn.functional as F
from torch.nn import Module
import math
import numpy as np
import os
import sys
dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(dir)
sys.path.append(root)
from ops.point_transform import transform_3D, transform_3D_to_2D, transform_2D


def gen_2D_gaussion_map(joint_2d, H, W, fx, fy, sigma):
    """

    :param joint_2d: Tensor(B, J, 3)
    :param H:
    :param W:
    :param sigma:
    :return: Tensor(B, J, H, W)
    """
    B, J, _ = joint_2d.shape
    u = torch.arange(W, device=joint_2d.device)
    v = torch.arange(H, device=joint_2d.device)
    v_t, u_t = torch.meshgrid([v, u]) # (H, W)
    grid = torch.stack([u_t, v_t], dim=-1)[None, ...].repeat([B, 1, 1, 1]) # (B, H, W, 2)
    grid = grid.reshape([B, H*W, 2])
    grid = grid[:, None, :, :].repeat([1, J, 1, 1]) # (B, J, H*W, 2)
    grid = grid.float() + 0.5 # coordinate of pixel is on center of pixel
    joint_2d = joint_2d[:, :, None, :].repeat([1, 1, H*W, 1]) # (B, J, W*H, 2)
    scale = joint_2d[:, :, :, 2]
    diff_x = ((grid[..., 0] - joint_2d[..., 0]) * scale / fx) ** 2
    diff_y = ((grid[..., 1] - joint_2d[..., 1]) * scale / fy) ** 2
    # diff_x = (grid[..., 0] - joint_2d[..., 0]) ** 2
    # diff_y = (grid[..., 1] - joint_2d[..., 1]) ** 2
    diff = diff_x + diff_y
    gaussian_map = 1 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-diff/(2*(sigma ** 2)))
    gaussian_map = gaussian_map.reshape([B, J, H, W])
    return gaussian_map


class LossCalculator(Module):
    def forward(self, heatmaps, joint_2d_pred, joint_3d_pred, view_trans, crop_trans, fx, fy, u0, v0, joint_3d_gt):
        """

        :param heatmap: Tensor(B, N, nstack, J, H, W)
        :param joint_2d_pred: Tensor(B, N, J, 2)
        :param joint_3d_pred: Tensor(B, J, 3)
        :param view_trans: Tensor(B, N, 4, 4)
        :param crop_trans: Tensor(B, 3, 3)
        :param fx: float
        :param fy: float
        :param u0: float
        :param v0: float
        :param joint_3d_gt: Tensor(B, J, 3)
        :return:
        """
        B, N, nstack, J, H, W = heatmaps.shape
        joint_3d_gt_expand = joint_3d_gt[:, None, :, :].repeat([1, N, 1, 1]) # (B, N, J, 3)
        crop_trans = crop_trans[:, None, :, :].repeat([1, N, 1, 1]) # (B, N, 3, 3)
        joint_3d_gt_expand = joint_3d_gt_expand.reshape([B * N, J, 3]) # (B*N, J, 3)
        view_trans = view_trans.reshape([B * N, 4, 4])
        crop_trans = crop_trans.reshape([B * N, 3, 3])
        joint_3d_gt_expand = transform_3D(joint_3d_gt_expand, view_trans) # (B*N, J, 3)
        joint_2d_gt = transform_3D_to_2D(joint_3d_gt_expand, fx, fy, u0, v0) # (B*N, J, 2)
        joint_2d_gt_crop = transform_2D(joint_2d_gt, crop_trans) / 4.
        heatmaps = heatmaps.reshape([B * N * nstack, J, H, W])
        gaussian_maps = gen_2D_gaussion_map(joint_2d_gt_crop, H, W, fx, fy, sigma=0.4) # (B*N, J, H, W)
        gaussian_maps = gaussian_maps[:, None, :, :, :].repeat([1, nstack, 1, 1, 1]).reshape([B*N*nstack, J, H, W])
        hm_loss = F.mse_loss(heatmaps, gaussian_maps, reduction='none')
        hm_loss = hm_loss.reshape([B, -1]).mean(-1)

        joint_2d_pred = joint_2d_pred.reshape([B*N, J, 2])
        error_2d = torch.norm(joint_2d_pred-joint_2d_gt[..., :2], dim=-1).mean(-1).reshape([B, N]).mean(-1)
        error_3d = torch.norm(joint_3d_pred - joint_3d_gt, dim=-1).mean(-1)
        return hm_loss, error_2d, error_3d, gaussian_maps.reshape([B, N, nstack, J, H, W])


class MultiA2JCalculator(Module):
    def __init__(self, reg_factor, conf_factor):
        super().__init__()
        self.reg_factor = reg_factor
        self.conf_factor = conf_factor
        self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')

    def forward(self, anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm, joints_3d_pred,
                joints_3d_fused, joint_3d_conf, view_trans, crop_trans, com_2d, cube, fx, fy, u0, v0, joints_3d_gt):
        """
        :param anchor_joints_2d_crop: Tensor(B, N, J, 2)
        :param regression_joints_2d_crop: Tensor(B, N, J, 2)
        :param depth_value_norm: Tensor(B, N, J)
        :param joint_3d_pred: Tensor(B, N, J, 3)
        :param joints_3d_fused: Tensor(B, J, 3)
        :param joint_3d_conf: Tensor(B, J, 3)
        :param view_trans: Tensor(B, N, 4, 4)
        :param crop_trans: Tensor(B, 3, 3)
        :param com_2d: Tensor(B, 3)
        :param cube: Tensor(B, 3)
        :param fx: float
        :param fy: float
        :param u0: float
        :param v0: float
        :param joints_3d_gt: Tensor(B, J, 3)
        :return:
        """
        B, N, J, _ = anchor_joints_2d_crop.shape
        joints_3d_gt_expand = joints_3d_gt[:, None, :, :].repeat([1, N, 1, 1]) # (B, N, J, 3)
        joints_3d_gt_expand = transform_3D(joints_3d_gt_expand, view_trans) # (B, N, J, 3)
        joints_2d_gt_expand = transform_3D_to_2D(joints_3d_gt_expand, fx, fy, u0, v0)[..., :2] # (B, N, J, 2)
        crop_trans_expand = crop_trans[:, None, :, :].repeat([1, N, 1, 1])
        joints_2d_gt_expand_crop = transform_2D(joints_2d_gt_expand, crop_trans_expand)
        com_z_expand = com_2d[:, None, :].repeat([1, N, 1])[:, :, 2:]
        cube_z_expand = cube[:, None, :].repeat([1, N, 1])[:, :, 2:]
        depth_gt_norm_expand = (joints_3d_gt_expand[..., 2]-com_z_expand)/(cube_z_expand/2)

        anchor_loss = self.smooth_l1_loss(anchor_joints_2d_crop, joints_2d_gt_expand_crop)
        regression_loss = self.smooth_l1_loss(regression_joints_2d_crop, joints_2d_gt_expand_crop)
        depth_loss = self.smooth_l1_loss(depth_value_norm, depth_gt_norm_expand)
        conf_loss = self.smooth_l1_loss(joint_3d_conf, joints_3d_gt)

        anchor_loss = anchor_loss.reshape([B, -1]).mean(-1)
        regression_loss = regression_loss.reshape([B, -1]).mean(-1)
        depth_loss = depth_loss.reshape([B, -1]).mean(-1)
        conf_loss = conf_loss.reshape([B, -1]).mean(-1)

        reg_loss = regression_loss*0.5 + depth_loss

        loss = anchor_loss + reg_loss * self.reg_factor + conf_loss*self.conf_factor

        error_3d = torch.norm(joints_3d_pred-joints_3d_gt[:, None, :, :], dim=-1).mean(-1)
        error_3d_fused = torch.norm(joints_3d_fused-joints_3d_gt, dim=-1).mean(-1)
        error_3d_conf = torch.norm(joint_3d_conf-joints_3d_gt, dim=-1).mean(-1)
        center_error_3d = error_3d[:, N//2]
        min_error_3d, _ = torch.min(error_3d, dim=-1)
        mean_error_3d = torch.mean(error_3d, dim=-1)
        return anchor_loss, reg_loss, conf_loss, loss, center_error_3d, min_error_3d, mean_error_3d, error_3d, \
               error_3d_fused, error_3d_conf


class ConfidenceLossCalculator(Module):
    def forward(self, confidence, joint_3d_pred, joint_3d_gt, view_trans, fx, fy, u0, v0, joint_2d_expand):
        '''

        :param confidence: Tensor(B, N)
        :param joint_3d_pred: Tensor(B, J, 3)
        :param joint_3d_gt: Tensor(B, J, 3)
        :param view_trans: Tensor(B, N, 4, 4)
        :param fx: float
        :param fy: flaat
        :param u0: float
        :param v0: float
        :param joint_2d_expand: Tensor(B, N, J, 2)
        :return:
        '''
        B, N = confidence.shape
        J = joint_3d_pred.shape[1]
        if N==3:
            map_shape = [1, 3]
        elif N==9:
            map_shape = [3, 3]
        elif N==15:
            map_shape = [3, 5]
        elif N==25:
            map_shape = [5, 5]
        elif N==81:
            map_shape = [9, 9]
        with torch.no_grad():
            error_3d = torch.norm(joint_3d_pred - joint_3d_gt, dim=-1).mean(-1)
        loss = F.smooth_l1_loss(joint_3d_pred, joint_3d_gt, reduction='none').mean(-1).mean(-1)
        confidence = confidence.reshape([B]+map_shape)
        return loss, error_3d, confidence

    def get_confidence(self, error_2d, map_shape):
        """

        :param error_2d: Tensor(B, J, N)
        :param map_shape: list
        :return:
            confidence: Tensor(B, N)
        """
        B, J, N = error_2d.shape
        error_std, error_mean = torch.std_mean(error_2d, dim=-1)
        error_std = error_std[:, :, None].repeat([1, 1, N])
        error_mean = error_mean[:, :, None].repeat([1, 1, N])
        confidence = -(error_2d - error_mean) / error_std
        soft_confidence = torch.softmax(confidence, dim=-1)
        # gauss_confidence = confidence
        # confidence = confidence.reshape([B*J, 1]+map_shape)
        # gauss_confidence = gaussian_blur2d(confidence, (5, 5), (1, 1))
        # gauss_confidence = gauss_confidence.reshape([B, J, N])
        # soft_gauss_confidence = torch.softmax(gauss_confidence, dim=-1)
        return soft_confidence


class ViewSelectLossCalculator(Module):
    def forward(self, light_heatmaps, heatmap, joint_3d_pred, view_trans, crop_trans, fx, fy, u0, v0, joint_3d_gt,
                alpha):
        """

        :param light_heatmaps: Tensor(B, N, nstack, J, H, W)
        :param heatmap: Tensor(B, N, J, H, W)
        :param joint_3d_pred: Tensor(B, J, 3)
        :param view_trans: Tensor(B, N, 4, 4)
        :param crop_trans: Tensor(B, 3, 3)
        :param fx: float
        :param fy: float
        :param u0: float
        :param v0: float
        :param joint_3d_gt: Tensor(B, J, 3)
        :return:
        """
        B, N, nstack, J, H, W = light_heatmaps.shape
        joint_3d_gt_expand = joint_3d_gt[:, None, :, :].repeat([1, N, 1, 1])  # (B, N, J, 3)
        crop_trans = crop_trans[:, None, :, :].repeat([1, N, 1, 1])  # (B, N, 3, 3)
        joint_3d_gt_expand = joint_3d_gt_expand.reshape([B * N, J, 3])  # (B*N, J, 3)
        view_trans = view_trans.reshape([B * N, 4, 4])
        crop_trans = crop_trans.reshape([B * N, 3, 3])
        joint_3d_gt_expand = transform_3D(joint_3d_gt_expand, view_trans)  # (B*N, J, 3)
        joint_2d_gt = transform_3D_to_2D(joint_3d_gt_expand, fx, fy, u0, v0)  # (B*N, J, 2)
        joint_2d_gt_crop = transform_2D(joint_2d_gt, crop_trans) / 4.
        if heatmap is not None:
            light_heatmaps = light_heatmaps.reshape([B * N * nstack, J, H, W])
            gaussian_maps = gen_2D_gaussion_map(joint_2d_gt_crop, H, W, fx, fy, sigma=0.4)  # (B*N, J, H, W)
            gaussian_maps = gaussian_maps[:, None, :, :, :].repeat([1, nstack, 1, 1, 1]).reshape(
                [B * N * nstack, J, H, W])
            heatmap = heatmap[:, :, None, :, :, :].repeat(1, 1, nstack, 1, 1, 1)
            heatmap = heatmap.reshape([B * N * nstack, J, H, W])
            hm_loss = alpha * F.mse_loss(light_heatmaps, gaussian_maps, reduction='none') + \
                      (1-alpha) * F.mse_loss(light_heatmaps, heatmap, reduction='none')
            hm_loss = hm_loss.reshape([B, -1]).mean(-1)
            gaussian_maps = gaussian_maps.reshape([B, N, nstack, J, H, W])
            gaussian_maps = gaussian_maps[:, :, 0, :, :, :]  # (B, N, J, H, W)
        else:
            hm_loss = torch.zeros([B], dtype=torch.float32, device=light_heatmaps.device)
            gaussian_maps = None
        error_3d = torch.norm(joint_3d_pred - joint_3d_gt, dim=-1).mean(-1)
        return hm_loss, error_3d, gaussian_maps


class ViewSelectA2JLossCalculator(Module):
    def __init__(self, alpha, conf_factor):
        super().__init__()
        self.alpha = alpha
        self.conf_factor = conf_factor
        self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')
        
    def forward(self, joints_3d_pred, joint_3d_fused, conf, joint_3d_conf_select,
                joints_3d_pred_select_light, joint_3d_fused_select_light, joint_3d_conf_select_light, conf_light,
                view_trans, crop_trans, com_2d, cube, fx, fy, u0, v0, joints_3d_gt):
        """
        :param joints_3d_pred: Tensor(B, N, J, 3)
        :param joint_3d_fused: Tensor(B, J, 3)
        :param conf: Tensor(B, N)
        :param joint_3d_conf_select: Tensor(B, J, 3)
        :param joints_3d_pred_select_light: Tensor(B, k, J, 3)
        :param joint_3d_fused_select_light: Tensor(B, J, 3)
        :param joint_3d_conf_select_light: Tensor(B, J, 3)
        :param conf_light: Tensor(B, N)
        :param crop_trans: Tensor(B, 3, 3)
        :param com_2d: Tensor(B, 3)
        :param cube: Tensor(B, 3)
        :param fx: float
        :param fy: float
        :param u0: float
        :param v0: float
        :param joints_3d_gt: Tensor(B, J, 3)
        :return:
        """
        B, N, J, _ = joints_3d_pred.shape
        joints_3d_gt_expand = joints_3d_gt[:, None, :, :].repeat([1, N, 1, 1])  # (B, N, J, 3)
        joints_3d_gt_expand = transform_3D(joints_3d_gt_expand, view_trans)  # (B, N, J, 3)
        joints_2d_gt_expand = transform_3D_to_2D(joints_3d_gt_expand, fx, fy, u0, v0)[..., :2]  # (B, N, J, 2)
        crop_trans_expand = crop_trans[:, None, :, :].repeat([1, N, 1, 1])
        joints_2d_gt_expand_crop = transform_2D(joints_2d_gt_expand, crop_trans_expand)
        com_z_expand = com_2d[:, None, :].repeat([1, N, 1])[:, :, 2:]
        cube_z_expand = cube[:, None, :].repeat([1, N, 1])[:, :, 2:]
        depth_gt_norm_expand = (joints_3d_gt_expand[..., 2] - com_z_expand) / (cube_z_expand / 2)

        # sub_conf = conf[:, :, None]-conf[:, None, :]
        # sub_conf_light = conf_light[:, :, None]-conf_light[:, None, :]
        # conf_loss = self.smooth_l1_loss(sub_conf*100, sub_conf_light*100)
        conf_loss = self.smooth_l1_loss(conf_light * self.conf_factor, conf * self.conf_factor)

        conf_loss = conf_loss.reshape([B, -1]).mean(-1)

        loss = conf_loss

        error_3d_fused_select_light = torch.norm(joint_3d_fused_select_light - joints_3d_gt, dim=-1).mean(-1)
        error_3d_conf_select_light = torch.norm(joint_3d_conf_select_light - joints_3d_gt, dim=-1).mean(-1)

        error_3d_fused = torch.norm(joint_3d_fused - joints_3d_gt, dim=-1).mean(-1)
        error_3d_conf_select = torch.norm(joint_3d_conf_select - joints_3d_gt, dim=-1).mean(-1)
        return conf_loss, loss, error_3d_fused_select_light, error_3d_conf_select_light,\
               error_3d_fused, error_3d_conf_select


class ViewSelectLossCalculator2(Module):
    def forward(self, joint_3d, joint_3d_uniform, confidence_select, joint_3d_gt):
        """

        :param joint_3d: Tensor(B, J, 3)
        :param joint_3d_uniform: Tensor(B, J, 3)
        :param confidence: Tensor(B, num_views)
        :param joint_3d_gt: Tensor(B, J, 3)
        :return:
        """
        error_3d = torch.norm(joint_3d - joint_3d_gt, dim=-1).mean(-1)
        error_3d_uniform = torch.norm(joint_3d_uniform - joint_3d_gt, dim=-1).mean(-1)
        # reward = torch.ones_like(error_3d)
        # reward[error_3d_uniform<error_3d] = -1
        reward = error_3d_uniform - error_3d
        weight = reward.clone()
        weight[reward < 0] = weight[reward < 0] * 0.5
        # reward[reward > 0] = reward[reward > 0] * 10
        # loss = torch.mean(-torch.log(confidence_select.mean(dim=-1))*reward, dim=-1) + 1e-3*error_3d/error_3d_uniform
        loss = -torch.log(confidence_select.sum(dim=-1)) * weight
        return loss, error_3d, error_3d_uniform, reward


class FocalLoss_Ori(Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=[0.25,0.75], gamma=2, balance_index=-1):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float,int)):
            assert 0<self.alpha<1.0, 'alpha should be in `(0,1)`)'
            assert balance_index >-1
            alpha = torch.ones((self.num_class))
            alpha *= 1-self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha,torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, pred, target):
        B = logit.size(0)
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous() # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1)) # [N,d1*d2..,C]-> [N*d1*d2..,C]
        p = torch.softmax(logit, -1)
        target = target.view(-1, 1) # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = p.gather(1, target).view(-1) + self.eps # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            alpha = self.alpha.to(logpt.device)
            alpha_class = alpha.gather(0,target.view(-1))
            logpt = alpha_class*logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt
        loss = torch.reshape(loss, [B, -1]).mean(dim=-1)
        target = target.reshape([B, -1])
        pred = pred.reshape([B, -1])
        acc = torch.sum(target==pred, dim=-1, dtype=torch.float32)/target.shape[-1]
        return loss, acc

if __name__ == '__main__':
    fx, fy, u0, v0 = 588.03, 587.07, 320., 320.
    # joint_2d = np.array([[0.5, 0.5, 500],
    #                      [0.5, 1.5, 500],
    #                      [0.5, 2.5, 500]], dtype=np.float32)
    # joint_2d = torch.from_numpy(joint_2d[None, :, :])
    # gaussian = gen_2D_gaussion_map(joint_2d, 3, 5, 1)
    # print(gaussian)

    # B, N, nstack, J, H, W = 8, 40, 4, 14, 64, 64
    # heatmaps = torch.rand((B, N, nstack, J, H, W), dtype=torch.float32).cuda()
    # joint_2d_pred = torch.rand((B, N, J, 2), dtype=torch.float32).cuda()
    # joint_3d_pred = torch.rand((B, J, 3), dtype=torch.float32).cuda()
    # view_trans = torch.rand((B, N, 4, 4), dtype=torch.float32).cuda()
    # crop_trans = torch.rand((B, 3, 3), dtype=torch.float32).cuda()
    # joint_3d_gt = torch.rand((B, J, 3), dtype=torch.float32).cuda()
    # loss_calc = LossCalculator()
    # loss_calc = torch.nn.DataParallel(loss_calc)
    # loss_calc = loss_calc.cuda()
    # hm_loss, error_2d, error_3d, gaussian_maps = loss_calc(heatmaps, joint_2d_pred, joint_3d_pred, view_trans, crop_trans, fx, fy, u0, v0, joint_3d_gt)
    # print(hm_loss.shape)
    # print(error_2d.shape)
    # print(error_3d.shape)
    # print(gaussian_maps.shape)

    # B = 8
    # N = 25
    # J = 14
    # error_2d_pred = torch.rand([B, N, J], dtype=torch.float32)
    # joint_2d_pred = torch.rand([B, N, J, 2], dtype=torch.float32)
    # joint_3d_gt = torch.rand([B, J, 3], dtype=torch.float32)
    # view_trans = torch.rand([B, N, 4, 4], dtype=torch.float32)
    #
    # error2d_loss_calc = ConfidenceLossCalculator()
    # error, loss, error_map_gt, error_map_pred = error2d_loss_calc(error_2d_pred, joint_2d_pred, joint_3d_gt, view_trans, fx, fy, u0, v0)
    # print(error.shape)
    # print(loss.shape)
    # print(error_map_gt.shape)
    # print(error_map_pred.shape)

    logit = torch.rand([4, 2, 480, 640], dtype=torch.float32)
    pred = torch.zeros([4, 480, 640], dtype=torch.int64)
    label = torch.zeros([4, 480, 640], dtype=torch.int64)
    loss_calc = FocalLoss_Ori(2)
    loss, acc = loss_calc(logit, pred, label)
    print(loss.shape)
    print(acc)
    print(acc.dtype)
