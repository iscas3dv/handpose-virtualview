import torch
import torch.nn.functional as F
import numpy as np


def normalize_depth(depth, com_2d, cube):
    """Normalize depth to [-1, 1]

    :param depth: (B, 1, H, W)
    :param com_2d: (B, 3)
    :param cube_z: float
    :return:
    """
    B, _, H, W = depth.shape
    background = (depth<1e-3).float()
    com_z = com_2d[:, 2]
    com_z = com_z[:, None, None, None].repeat((1, 1, H, W))
    cube_z = cube[:, 2]
    cube_z = cube_z[:, None, None, None].repeat((1, 1, H, W))
    norm_depth = depth + background * (com_z + (cube_z / 2.))
    norm_depth = (norm_depth-com_z) / (cube_z/2.)
    return norm_depth


def normalize_depth_expand(depth_expand, com_2d, cube):
    """Normalize depth expand to [-1, 1]

        :param depth: (B, num_views, 1, H, W)
        :param com_2d: (B, 3)
        :param cube_z: (B, 3)
        :return:
    """
    B, N, _, H, W = depth_expand.shape
    background = (depth_expand<1e-3).float()
    com_z = com_2d[:, 2]
    com_z = com_z[:, None, None, None, None].repeat((1, N, 1, H, W))
    cube_z = cube[:, 2]
    cube_z = cube_z[:, None, None, None, None].repeat((1, N, 1, H, W))
    norm_depth_expand = depth_expand + background * (com_z + (cube_z / 2.))
    norm_depth_expand = (norm_depth_expand - com_z) / (cube_z / 2.)
    return norm_depth_expand


def normalize_image(img):
    """

    :param img: Tensor(B, 1, H, W)
    :return: Tensor(B, 1, H, W)
    """
    B, _, H, W = img.shape
    t_min, _ = torch.min(img.reshape([B, -1]), dim=-1)
    t_max, _ = torch.max(img.reshape([B, -1]), dim=-1)
    t_min = t_min[:, None].repeat(1, H*W).reshape([B, 1, H, W])
    t_max = t_max[:, None].repeat(1, H*W).reshape([B, 1, H, W])
    img = (img-t_min)/(t_max-t_min)
    return img


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
sobel_x = sobel_x.reshape((1, 1, 3, 3))
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
sobel_y = sobel_y.reshape((1, 1, 3, 3))
pad = torch.nn.ReplicationPad2d(1)

def sobel_edge(img):
    """

    :param img: Tensor(B, 1, H, W)
    :return:
    """
    weight_x = torch.tensor(sobel_x, device=img.device, requires_grad=False)
    weight_y = torch.tensor(sobel_y, device=img.device, requires_grad=False)
    img = pad(img)
    edge_x = F.conv2d(img, weight_x)
    edge_y = F.conv2d(img, weight_y)
    edge = torch.abs(edge_x) + torch.abs(edge_y)
    # edge = torch.sqrt(edge_x*edge_x+edge_y*edge_y)
    return edge


def normalize_edge(edge: torch.Tensor):
    """

    :param edge: Tensor(B, 1, H, W)
    :return:
    """
    B, _, H, W = edge.size()
    edge = edge.reshape([B, -1])
    torch.min(edge, 1, keepdim=True)
    t_min = torch.min(edge, 1, keepdim=True)[0].repeat([1, H*W])
    t_max = torch.max(edge, 1, keepdim=True)[0].repeat([1, H*W])
    edge = (edge-t_min) / (t_max-t_min)
    edge = edge.reshape([B, 1, H, W])
    return edge


if __name__ == '__main__':
    B, N, H, W = 4, 12, 128, 128
    depth_expand = torch.randn((B, N, 1, H, W)).cuda()
    com_2d = torch.randn((B, 3)).cuda()
    cube_z = 125.
    output = normalize_depth_expand(depth_expand, com_2d, cube_z)
    print(output.shape)