import torch


def transform_2D_to_3D(points, fx, fy, u0, v0):
    """

    :param points: Tensor(..., 3)
    :param fx:
    :param fy:
    :param u0:
    :param v0:
    :return: Tensor(..., 3)
    """
    x = (points[..., 0] - u0) * points[..., 2] / fx
    y = (points[..., 1] - v0) * points[..., 2] / fy
    z = points[..., 2]
    return torch.stack([x, y, z], dim=-1)


def transform_3D_to_2D(points, fx, fy, u0, v0):
    u = points[..., 0] / points[..., 2] * fx + u0
    v = points[..., 1] / points[..., 2] * fy + v0
    d = points[..., 2]
    return torch.stack([u, v, d], dim=-1)


def transform_3D(points, trans_matrix):
    """3D affine transformation

    :param points: Tensor(..., N, 3)
    :param trans_matrix: Tensor(..., 4, 4)
    :return: Tensor(B, N, 3)
    """
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    ones = torch.ones_like(x)
    points_h = torch.stack([x, y, z, ones], -2) # (..., 4, N)
    points_h = trans_matrix @ points_h
    points = torch.transpose(points_h, -2, -1)[..., :3] # (..., N, 3)
    return points


def transform_2D(points, trans_matirx):
    """2D affine transformation

    :param points: Tensor(..., N, 2|3)
    :param trans_matirx: Tensor(..., 3, 3)
    :return: Tensor(..., N, 2|3)
    """
    d = points.size(-1)
    x = points[..., 0]
    y = points[..., 1]
    if d > 2:
        z = points[..., 2]
    ones = torch.ones_like(x)
    points_h = torch.stack([x, y, ones], axis=-2) # (B, 3, N)
    points_h = trans_matirx @ points_h
    points = torch.transpose(points_h, -2, -1) # (B, N, 3)
    if d > 2:
        points[..., 2] = z
    else:
        points = points[..., :2]
    return points


def transform(points, trans_matrix):
    """2D or 3D affine transformation.
    This function is the same as the function of the above two functions.
    But it can backward.

    :param points: Tensor(B, N, 3)
    :param trans_matrix: Tensor(B, 3/4, 3/4)
    :return: Tensor(B, N, 3)
    """
    # B, N, _ = points.shape
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    ones = torch.ones_like(x, requires_grad=False)
    if trans_matrix.size(1)==4:
        # points = points.transpose(1, 2)
        # points[:, :3, :] = trans_matrix[:, :3, :3] @ points[:, :3, :]
        # points[:, :3, :] += trans_matrix[:, :3, 3, None].repeat([1, 1, N])
        # points = points.transpose(1, 2)
        points_h = torch.stack([x, y, z, ones], -2)
        points_h = trans_matrix @ points_h
        points = torch.transpose(points_h, 1, 2)[..., :3]
    elif trans_matrix.size(1)==3:
        # points = points.transpose(1, 2)
        # points[:, :2, :] = trans_matrix[:, :2, :2] @ points[:, :2, :]
        # points[:, :2, :] += trans_matrix[:, :2, 2, None].repeat([1, 1, N])
        # points = points.transpose(1, 2)
        points_h = torch.stack([x, y, ones], -2)
        points_h = trans_matrix @ points_h
        points = torch.transpose(points_h, 1, 2)
        points[..., 2] = z
    return points
