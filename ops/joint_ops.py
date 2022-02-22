import torch
import numpy as np
import time
import os
import sys
dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(dir)
sys.path.append(root)
from ops.loss_ops import gen_2D_gaussion_map


def solve(coor, b):
    """Ax=b, solve x, where
    A = [[coor[?, 0]**2, coor[?, 0], 1],
         [coor[?, 1]**2, coor[?, 1], 1],
         [coor[?, 2]**2, coor[?, 2], 1]]
    x is (a, b, c) which is coefficient of quadratic function,

    :param coor: Tensor(B, 3)
    :param b: Tensor(B, 3)
    :return: Tensor(B) -b/(a*2), the symmetry axis of quadratic function

    """
    B = coor.shape[0]
    A = torch.ones((B, 3, 3), dtype=torch.float32, device=b.device)
    A[:, :, 1] = coor
    A[:, :, 0] = A[:, :, 1]*A[:, :, 1]
    U, D, V = torch.svd(A) # (B, 3, 3), (B, 3), (B, 3, 3)
    b_ = U.transpose(-2, -1) @ b[:, :, None] # (B, 3, 1)
    b_ = b_.squeeze() # (B, 3)
    y = b_/D # (B, 3)
    result = V @ y[:, :, None] # (B, 3, 1)
    result = result.squeeze(dim=-1) # (B, 3)
    not_zero = (result[:, 0]!=0)
    x = coor[:, 1].clone()
    x[not_zero] = -result[not_zero, 1] / (result[not_zero, 0] * 2)
    return x


def heatmap_to_loc(heatmap, adjust=True):
    """

    :param heatmap: Tensor(B, num_joints, H, W)
    :return: Tensor(B, num_joints, 2)
    """
    device = heatmap.device
    B, J, H, W = heatmap.shape
    heatmap = heatmap.reshape((B*J, H, W))
    dense_flat = heatmap.reshape((B*J, -1))
    loc = torch.argmax(dense_flat, dim=-1)
    y = loc//W # (B*num_joints)
    x = loc%W # (B*num_joints)
    xx = x.float()+0.5
    yy = y.float()+0.5

    # adjust location. It is extremely slow on GPU, so we use CPU
    if adjust:
        x, y, xx, yy, dense_flat, loc = x.cpu(), y.cpu(), xx.cpu(), yy.cpu(), dense_flat.cpu(), loc.cpu()
        x_adjust_index = (0<x) & (x<W-1)
        if torch.any(x_adjust_index):
            coor_x = xx[x_adjust_index, None].repeat([1, 3])
            coor_x[:, 0] -= 1.
            coor_x[:, 2] += 1.
            b_x = torch.zeros_like(coor_x)
            b_x[:, 0] = dense_flat[x_adjust_index, loc[x_adjust_index]-1]
            b_x[:, 1] = dense_flat[x_adjust_index, loc[x_adjust_index]]
            b_x[:, 2] = dense_flat[x_adjust_index, loc[x_adjust_index]+1]
            xx[x_adjust_index] = solve(coor_x, b_x)
            # adjust = torch.zeros_like(coor_x[:, 1])
            # adjust[b_x[:, 0]<b_x[:, 2]] = 0.25
            # adjust[b_x[:, 0]>b_x[:, 2]] = -0.25
            # adjust_coor_x = coor_x[:, 1] + adjust
            # xx[x_adjust_index] = adjust_coor_x

        y_adjust_index = (0 < y) & (y < H - 1)
        if torch.any(y_adjust_index):
            coor_y = yy[y_adjust_index, None].repeat([1, 3]).float()
            coor_y[:, 0] -= 1.
            coor_y[:, 2] += 1.
            b_y = torch.zeros_like(coor_y)
            b_y[:, 0] = dense_flat[y_adjust_index, loc[y_adjust_index] - W]
            b_y[:, 1] = dense_flat[y_adjust_index, loc[y_adjust_index]]
            b_y[:, 2] = dense_flat[y_adjust_index, loc[y_adjust_index] + W]
            yy[y_adjust_index] = solve(coor_y, b_y)
            # adjust = torch.zeros_like(coor_y[:, 1])
            # adjust[b_y[:, 0] < b_y[:, 2]] = 0.25
            # adjust[b_y[:, 0] > b_y[:, 2]] = -0.25
            # adjust_coor_y = coor_y[:, 1] + adjust
            # yy[y_adjust_index] = adjust_coor_y

        xx, yy = xx.to(device), yy.to(device)

        # for b in range(B):
        #     for j in range(J):
        #         ax, ay = x[b, j], y[b, j]
        #         tmp = heatmap[b, j]
        #         # if (ax, ay) is not on bound
        #         if 0<ax and ax<W-1:
        #             # if tmp[ay, ax-1]<tmp[ay, ax+1]:
        #             #     xx[b, j] += 0.25
        #             # elif tmp[ay, ax-1]>tmp[ay, ax+1]:
        #             #     xx[b, j] -= 0.25
        #             xx[b, j] = solve(torch.stack([ax-0.5, ax+0.5, ax+1.5]), tmp[ay, ax-1:ax+2])
        #
        #         if 0<ay and ay<H-1:
        #             # if tmp[ay-1, ax]<tmp[ay+1, ax]:
        #             #     yy[b, j] += 0.25
        #             # elif tmp[ay-1, ax]>tmp[ay+1, ax]:
        #             #     yy[b, j] -= 0.25
        #             yy[b, j] = solve(torch.stack([ay-0.5, ay+0.5, ay+1.5]), tmp[ay-1:ay+2, ax])
    xx = xx.reshape([B, J])
    yy = yy.reshape([B, J])
    return torch.stack([xx, yy], dim=-1)


def get_projection_matrices(inter_matrix, view_trans):
    """

    :param inter_matrix: Tensor(B, 3, 3)
    :param view_trans: Tensor(B, num_views, 4, 4)
    :return: Tensor(B, num_views, 3, 4)
    """
    B, N = view_trans.size(0), view_trans.size(1)
    eye = torch.eye(3, 4, dtype=torch.float32, device=inter_matrix.device)
    eye = eye[None, ...].repeat([B, 1, 1]) # (B, 3, 4)
    proj_mat = inter_matrix@eye # (B, 3, 4)
    proj_mat = proj_mat[:, None, :, :].repeat(1, N, 1, 1) # (B, num_views, 3, 4)
    proj_mat = proj_mat @ view_trans # (B, num_views, 3, 4)
    return proj_mat


def triangulate(joint_2d, cam_mat, weight=None):
    """

    :param joint_2d: Tensor(B, num_joints, num_views, 2)
    :param cam_mat: Tensor(B, num_joints, num_views, 3, 4)
    :param weight: Tensor(B, num_joints, num_views)
    :return:
    """
    B, J, N, _ = joint_2d.shape
    joint_2d = joint_2d[..., None] # (B, num_joints, num_views, 2, 1)
    c2 = cam_mat[..., 2:, :] # (B, num_joints, num_views, 1, 4)
    c12 = cam_mat[..., :2, :] # (B, num_joints, num_views, 2, 4)
    A = joint_2d @ c2 - c12 # (B, num_joints, num_views, 2, 4)
    if weight is not None:
        weight = weight[:, :, :, None, None].repeat([1, 1, 1, 2, 4])
        A = A * weight
    A = A.reshape([B, J, N*2, 4]) # (B, num_joints, num_views*2, 4)
    device = A.device
    A = A.cpu()
    _, _, V = torch.svd(A) # (B, num_joints, 4, 4)
    V = V.to(device)
    X = V[..., -1] # (B, num_joints, 4)
    X = X / X[..., -1, None] # (B, num_joints, 4)
    joint_3d = X[..., :-1] # (B, num_joints, 3)
    return joint_3d


def compute_joint_3d(joint_2d, inter_matrix, view_trans, weight=None):
    """Calculate 3D joint according to 2D joint.

    :param joint_2d: Tensor(B, num_views, num_joints, 2)
    :param inter_matrix: Tensor(B, 3, 3)
    :param view_trans: Tensor(B, num_views, 4, 4)
    :param weight: Tensor(B, num_joints, num_views)
    :return: Tensor(B, J, 3)
    """
    J = joint_2d.size(2)
    proj_mat = get_projection_matrices(inter_matrix, view_trans) # (B, num_views, 3, 4)
    joint_2d = joint_2d.permute([0, 2, 1, 3]) # (B, num_joints, num_views, 2)
    proj_mat = proj_mat[:, None, ...].repeat([1,J, 1, 1, 1]) # (B, num_joints, num_views, 3, 4)
    joint_3d = triangulate(joint_2d, proj_mat, weight)
    return joint_3d


def compute_joint_3d_view_select(confidence, joint_2d_pred, inter_matrix, view_trans):
    """

    :param confidence: Tensor(B, N)
    :param joint_2d_pred: Tensor(B, N, J, 2)
    :param inter_matrix: Tensor(B, 3, 3)
    :param view_trans: Tensor(B, N, 4, 4)
    :return:
        joint_3d_select: Tensor(B, J, 3)
    """
    B, N, J, _ = joint_2d_pred.shape
    indices = torch.multinomial(confidence, 10, replacement=False)
    joint_2d_indices = indices[:, :, None, None].repeat([1, 1, J, 2])
    joint_2d_select = torch.gather(joint_2d_pred, 1, joint_2d_indices)
    view_trans_indices = indices[:, :, None, None].repeat([1, 1, 4, 4])
    view_trans_select = torch.gather(view_trans, 1, view_trans_indices)

    # _, indices = torch.sort(confidence, dim=-1, descending=True)
    # joint_2d_indices = indices[:, :, None, None].repeat([1, 1, J, 2])
    # joint_2d_select = joint_2d_pred.reshape([-1])[joint_2d_indices.reshape(-1)<10].reshape([B, 10, J, 2])
    # view_trans_indices = indices[:, :, None, None].repeat([1, 1, 4, 4])
    # view_trans_select = view_trans.reshape([-1])[view_trans_indices.reshape(-1)<10].reshape([B, 10, 4, 4])

    joint_3d_select = compute_joint_3d(joint_2d_select, inter_matrix, view_trans_select)
    return joint_3d_select


if __name__ == '__main__':
    # x = torch.from_numpy(np.array([1., 1., 1.], dtype=np.float32))
    # b = torch.from_numpy(np.array([3., 3., 3.], dtype=np.float32))
    # print(solve(x, b))
    H = W = 32
    B = 480
    joint_2d = torch.rand((B, 14, 3), dtype=torch.float32)*32
    joint_2d[:, :, -1] = 1.
    # joint_2d = torch.ones((B, 14, 2), dtype=torch.float32)
    # joint_2d = torch.ones((B, 14, 2), dtype=torch.float32)
    # joint_2d[:, :, 1] = torch.rand((B, 14), dtype=torch.float32)*32
    # joint_2d[:, :, 0] = torch.rand((B, 14), dtype=torch.float32) * 32
    print(joint_2d[0])
    joint_2d = joint_2d.cuda()

    heatmap = gen_2D_gaussion_map(joint_2d, H, W, 1, 1, 1)
    # print(heatmap.shape)
    # heatmap = torch.zeros((1, 2, 32, 32), dtype=torch.float32)
    # heatmap[:, :, 5:8, 5:8] = 1
    split = time.time()
    loc = heatmap_to_loc(heatmap)
    print(loc[0])
    print(time.time() - split)

    split = time.time()
    loc = heatmap_to_loc(heatmap, False)
    print(loc[0])
    print(time.time() - split)
