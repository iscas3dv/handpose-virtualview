import torch
import numpy as np
import depth_to_point_cloud_mask_cuda
import point_cloud_mask_to_depth_cuda
import sys
import os
dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(dir)
sys.path.append(root)
from torch.utils.data import DataLoader
from ops.point_transform import transform_3D, \
    transform_2D_to_3D, transform_3D_to_2D, transform_2D
from feeders.nyu_feeder import NyuFeeder
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s %(name)s:%(lineno)d] %(message)s")
logger = logging.getLogger(__file__)


# def crop_trans_inv(crop_trans):
#     '''
#
#     :param crop_trans: Tensor(..., 3, 3)
#     :return:
#     '''
#     inv = torch.zeros_like(crop_trans)
#     inv[..., 0, 0] = 1 / crop_trans[..., 0, 0]
#     inv[..., 0, 2] = -crop_trans[..., 0, 2] / crop_trans[..., 0, 0]
#     inv[..., 1, 1] = 1 / crop_trans[..., 1, 1]
#     inv[..., 1, 2] = -crop_trans[..., 1, 2] / crop_trans[..., 1, 1]
#     inv[..., 2, 2] = 1.
#
#     return inv


def depth_to_point_cloud_mask(depth):
    """

    :param depth: Tensor(B, 1, H, W)
    :return: point_cloud: Tensor(B, N, 3), mask: Tensor(B, N)
    """
    depth = depth.permute((0, 2, 3, 1)) # (B, H, W, 1)
    return depth_to_point_cloud_mask_cuda.forward(depth.contiguous())


def point_cloud_mask_to_depth(point_cloud, mask, h, w):
    depth = point_cloud_mask_to_depth_cuda.forward(point_cloud.contiguous(), mask, h, w) # (B, H, W, 1)
    depth = depth.permute((0, 3, 1, 2)) # (B, 1, H, W)
    return depth


def uniform_view_matrix(center, level, random_sample, random_rotate):
    """Uniform generation of view transformation matrix

    :param center: Tensor(B, 3), 3D coordinate
    :param level: int, 1, 2, 3, 4 or 5
    :return: Tensor(B, num_views, 4, 4)
    """
    B = center.size(0)
    if random_sample:
        if level == 0:
            num_view = 1
        elif level == 1:
            num_view = 3
        elif level == 2:
            num_view = 9
        elif level == 3:
            num_view = 15
        elif level == 4:
            num_view = 25
        elif level == 5:
            num_view = 81
        else:
            logger.critical('level must be 1, 2, 3 or 4.')
            raise ValueError('level must be 1, 2, 3 or 4.')
        rotation = torch.from_numpy(np.random.uniform(-np.pi/3, np.pi/3, size=[num_view, 2])).to(center.device)
        rotation = rotation.float()
    else:
        if level == 1:
            # azimuth = torch.arange(-np.pi / 3, np.pi / 3 + 0.01, np.pi / 3, device=center.device)  # 3
            azimuth = torch.linspace(-np.pi / 3, np.pi / 3, 3, device=center.device)
            elevation = torch.zeros([1], device=center.device)
        elif level == 2:
            # azimuth = torch.arange(-np.pi / 3, np.pi / 3 + 0.01, np.pi / 3, device=center.device)  # 3
            # elevation = torch.arange(-np.pi / 3, np.pi / 3 + 0.01, np.pi / 3, device=center.device)  # 3
            azimuth = torch.linspace(-np.pi / 3, np.pi / 3, 3, device=center.device) # 3
            elevation = torch.linspace(-np.pi / 3, np.pi / 3, 3, device=center.device)  # 3
        elif level == 3:
            # azimuth = torch.arange(-np.pi / 3, np.pi / 3 + 0.01, np.pi / 6, device=center.device)  # 5
            # elevation = torch.arange(-np.pi / 3, np.pi / 3 + 0.01, np.pi / 3, device=center.device)  # 3
            azimuth = torch.linspace(-np.pi / 3, np.pi / 3, 5, device=center.device)  # 5
            elevation = torch.linspace(-np.pi / 3, np.pi / 3, 3, device=center.device)  # 3
        elif level == 4:
            # azimuth = torch.arange(-np.pi / 3, np.pi / 3 + 0.01, np.pi / 6, device=center.device)  # 5
            # elevation = torch.arange(-np.pi / 3, np.pi / 3 + 0.01, np.pi / 6, device=center.device)  # 5
            azimuth = torch.linspace(-np.pi / 3, np.pi / 3, 5, device=center.device)  # 5
            elevation = torch.linspace(-np.pi / 3, np.pi / 3, 5, device=center.device)  # 5
        elif level == 5:
            # azimuth = torch.arange(-np.pi / 3, np.pi / 3 + 0.01, np.pi / 12, device=center.device)  # 9
            # elevation = torch.arange(-np.pi / 3, np.pi / 3 + 0.01, np.pi / 12, device=center.device)  # 9
            azimuth = torch.linspace(-np.pi / 3, np.pi / 3, 9, device=center.device)  # 9
            elevation = torch.linspace(-np.pi / 3, np.pi / 3, 9, device=center.device)  # 9
        else:
            logger.critical('level must be 1, 2, 3 or 4.')
            raise ValueError('level must be 1, 2, 3 or 4.')

        elevation = elevation.float()
        azimuth = azimuth.float()

        rotation = torch.meshgrid(elevation, azimuth)
        rotation = torch.reshape(torch.stack(rotation, axis=-1), [-1, 2])

    rotation = rotation[None, :, :].repeat(B, 1, 1)
    # print(rotation)

    N = rotation.size(1)
    r_theta_x = rotation[..., 0]
    r_theta_y = rotation[..., 1]
    if random_rotate:
        # r_theta_z = torch.rand([B, rotation.shape[1]], dtype=torch.float32, device=center.device) * np.pi * 2
        r_theta_z = torch.ones([B, rotation.shape[1]], dtype=torch.float32, device=center.device) * np.pi * 2 * \
                    np.random.rand()
    else:
        r_theta_z = torch.zeros([B, rotation.shape[1]], dtype=torch.float32, device=center.device)
    center = center.float()
    transform_center = center[:, None, :].repeat(1, N, 1)
    zeros = torch.zeros([B, N], dtype=torch.float32, device=center.device)
    ones = torch.ones([B, N], dtype=torch.float32, device=center.device)

    c, s = torch.cos(r_theta_x), torch.sin(r_theta_x)
    Rx = torch.stack([ones, zeros, zeros, zeros,
                      zeros, c, -s, zeros,
                      zeros, s, c, zeros,
                      zeros, zeros, zeros, ones], axis=-1)
    Rx = torch.reshape(Rx, [B, N, 4, 4])

    c, s = torch.cos(r_theta_y), torch.sin(r_theta_y)
    Ry = torch.stack([c, zeros, s, zeros,
                      zeros, ones, zeros, zeros,
                      -s, zeros, c, zeros,
                      zeros, zeros, zeros, ones], axis=-1)
    Ry = torch.reshape(Ry, [B, N, 4, 4])

    c, s = torch.cos(r_theta_z), torch.sin(r_theta_z)
    Rz = torch.stack([c, -s, zeros, zeros,
                      s, c, zeros, zeros,
                      zeros, zeros, ones, zeros,
                      zeros, zeros, zeros, ones], axis=-1)
    Rz = torch.reshape(Rz, [B, N, 4, 4])

    to_center = torch.stack([ones, zeros, zeros, -transform_center[..., 0],
                             zeros, ones, zeros, -transform_center[..., 1],
                             zeros, zeros, ones, -transform_center[..., 2],
                             zeros, zeros, zeros, ones], axis=-1)
    to_center = torch.reshape(to_center, [B, N, 4, 4])

    # to_center_inv = torch.stack([ones, zeros, zeros, transform_center[..., 0],
    #                              zeros, ones, zeros, transform_center[..., 1],
    #                              zeros, zeros, ones, transform_center[..., 2],
    #                              zeros, zeros, zeros, ones], axis=-1)
    # to_center_inv = torch.reshape(to_center_inv, [B, N, 4, 4])

    transform_mat = torch.inverse(to_center) @ Ry @ Rx @ Rz @ to_center
    return transform_mat


def depth_crop_expand(depth_crop, fx, fy, u0, v0, crop_trans, level, com_2d, random_sample, random_ratote=False,
                      indices=None):
    """When
    level=1, num_views=3
    level=2, num_views=9
    level=3, num_views=15
    level=4, num_views=25
    level=5, num_views=81

    :param depth_crop: Tensor(B, 1, H, W)
    :param fx: float
    :param fy: float
    :param u0: float
    :param v0: float
    :param crop_trans: Tensor(B, 3, 3)
    :param level: int, 1, 2, 3, 4, 5
    :param com_2d: Tensor(B, 3)
    :param random_sample: bool
    :param random_ratote: bool
    :param indices: Tensor(B, num_select)
    :return:
        if indices is None:
            depth_crop_expand: Tensor(B, num_views, 1, H, W)
            view_mat: Tensor(B, num_views, 4, 4)
        else:
            depth_crop_expand: Tensor(B, num_select, 1, H, W)
            view_mat: Tensor(B, num_select, 4, 4)
    """
    B, _, H, W = depth_crop.size()
    center = com_2d
    center = transform_2D_to_3D(center, fx, fy, u0, v0)
    view_mat = uniform_view_matrix(center, level, random_sample, random_ratote) # Tensor(B, num_views, 4, 4)
    if indices is None:
        num_views = view_mat.size(1)
    else:
        indices = indices[:, :, None, None].repeat([1, 1, 4, 4])
        view_mat = torch.gather(view_mat, 1, indices)
        num_views = indices.size(1)
    depth_crop = depth_crop[:, None, :, :, :].repeat([1, num_views, 1, 1, 1])
    depth_crop = depth_crop.reshape([B*num_views, 1, H, W])
    crop_trans = crop_trans[:, None, :, :].repeat([1, num_views, 1, 1])
    crop_trans = crop_trans.reshape([B*num_views, 3, 3])
    view_mat = view_mat.reshape([B*num_views, 4, 4])
    depth_expand = render_view(depth_crop, fx, fy, u0, v0, crop_trans, view_mat)

    depth_expand = depth_expand.reshape([B, num_views, 1, H, W])
    view_mat = view_mat.reshape([B, num_views, 4, 4])

    return depth_expand, view_mat

def render_view(depth_crop, fx, fy, u0, v0, crop_trans, view_mat):
    '''

    :param depth_crop: Tensor(B, 1, H, W)
    :param fx: float
    :param fy: float
    :param u0: float
    :param v0: float
    :param crop_trans: Tensor(B, 3, 3)
    :param view_mat: Tensor(B, 4, 4)
    :return: Tensor(B, 1, H, W)
    '''
    B, _, H, W = depth_crop.size()
    pc_crop, mask = depth_to_point_cloud_mask(torch.round(depth_crop).int())
    pc_crop, mask = pc_crop.float(), mask.float()
    pc = transform_2D(pc_crop, torch.inverse(crop_trans))
    pc_3d = transform_2D_to_3D(pc, fx, fy, u0, v0)
    pc_3d_trans = transform_3D(pc_3d, view_mat)
    pc_trans = transform_3D_to_2D(pc_3d_trans, fx, fy, u0, v0)
    pc_crop_trans = transform_2D(pc_trans, crop_trans)
    depth_expand = point_cloud_mask_to_depth(torch.round(pc_crop_trans).int(), mask.int(), H, W)
    return depth_expand.float()

if __name__ == '__main__':
    uniform_view_matrix(torch.zeros([1, 3]), level=4, random_sample=False, random_rotate=False)
    # from tqdm import tqdm
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # u0 = 320.0
    # v0 = 240.0
    # fx = 588.03
    # fy = 587.07
    # random_sample = True
    # import matplotlib.pyplot as plt
    # import utils.point_transform as np_pt
    # B = 4
    # train_dataset = NyuFeeder('train', max_jitter=10., depth_sigma=0., offset=30, random_flip=False)
    # dataloader = DataLoader(train_dataset, shuffle=False, batch_size=B, num_workers=8)
    # for batch_idx, batch_data in enumerate(tqdm(dataloader)):
    #     item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = batch_data
    #
    #     cropped, crop_trans, com_2d = cropped.cuda(), crop_trans.cuda(), com_2d.cuda()
    #     confidence = torch.ones([B, 25])
    #     indices = torch.multinomial(confidence, 3).cuda()
    #     crop_expand, view_mat = depth_crop_expand(cropped, fx, fy, u0, v0, crop_trans, 4, com_2d, random_sample=False,
    #                                               random_ratote=False, indices=indices)
    #     cropped = cropped.cpu().numpy()
    #     crop_trans = crop_trans.cpu().numpy()
    #     crop_expand = crop_expand.cpu().numpy()
    #     view_mat = view_mat.cpu().numpy()
    #     com_2d = com_2d.cpu().numpy()
    #     cube = cube.numpy()
    #     com_2d = com_2d[0]
    #     cube = cube[0]
        # plt.imshow(cropped[0, 0, ...])
        # plt.show()
        # print(crop_expand.shape)
        # for i in range(0, crop_expand.shape[1], 2):
        #     img = crop_expand[0, i, 0, ...]
        #     img[img>1e-3] = img[img>1e-3] - com_2d[2] + cube[2]/2.
        #     img[img<1e-3] = cube[2]
        #     img = img / cube[2]
        #     _joint_3d = joint_3d[0]
        #     _joint_3d = np_pt.transform_3D(_joint_3d, view_mat[0, i])
        #     _joint_2d = np_pt.transform_3D_to_2D(_joint_3d, fx, fy, u0, v0)
        #     _crop_joint_2d = np_pt.transform_2D(_joint_2d, crop_trans[0])
        #     fig, ax = plt.subplots(figsize=plt.figaspect(img))
        #     fig.subplots_adjust(0, 0, 1, 1)
        #     ax.imshow(img, cmap='gray')
        #     # ax.scatter(_crop_joint_2d[:, 0], _crop_joint_2d[:, 1], c='red', s=100)
        #     ax.axis('off')
        #     plt.savefig('{}.jpg'.format(i))
        #     plt.show()
        # break
