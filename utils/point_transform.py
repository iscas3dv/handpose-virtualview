import numpy as np


def transform_3D(points, trans_matrix):
    """3D affine transformation

    :param points: Tensor(..., N, 3)
    :param cam_trans_matrix: Tensor(..., N, 4)
    :return: Tensor(B, N, 3)
    """
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    ones = np.ones_like(x)
    points_h = np.stack([x, y, z, ones], -2) # (..., 4, N)
    points_h = trans_matrix @ points_h
    dim = len(points.shape)
    points = np.transpose(points_h, [i for i in range(dim-2)]+[dim-1,dim-2])[..., :3] # (..., N, 3)
    return points


def transform_3D_to_2D(points, fx, fy, u0, v0):
    u = points[..., 0] / points[..., 2] * fx + u0
    v = points[..., 1] / points[..., 2] * fy + v0
    d = points[..., 2]
    return np.stack([u, v, d], axis=-1)


def transform_2D_to_3D(points, fx, fy, u0, v0):
    x = (points[..., 0] - u0) * points[..., 2] / fx
    y = (points[..., 1] - v0) * points[..., 2] / fy
    z = points[..., 2]
    return np.stack([x, y, z], axis=-1)


def transform_2D(points, trans_matirx):
    """2D affine transformation

    :param points: Tensor(..., N, 3)
    :param trans_matirx: Tensor(..., 3, 3)
    :return: Tensor(..., N, 3)
    """
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    ones = np.ones_like(x)
    points_h = np.stack([x, y, ones], axis=-2) # (..., 4, N)
    points_h = trans_matirx @ points_h
    dim = len(points.shape)
    points = np.transpose(points_h, [i for i in range(dim-2)]+[dim-1,dim-2]) # (B, N, 3)
    points[..., 2] = z
    return points