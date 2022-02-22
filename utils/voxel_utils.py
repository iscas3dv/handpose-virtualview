import numpy as np


def gen_voxel(cropped, com_2d, cube, voxel_len):
    """

    :param cropped: numpy([H, W], float)
    :param com_2d: numpy([3], float)
    :param cube: numpy([3], float)
    :param voxel_len: int
    :return: numpy([voxel_len, voxel_len, voxel_len], int)
    """
    H, W = cropped.shape

    # Where x is the x row, y is the y column
    x = np.arange(H)
    y = np.arange(W)
    x, y = np.meshgrid(x, y, indexing='ij')
    z = cropped.copy()
    mask = np.bitwise_and(cropped>=com_2d[2]-cube[2]/2., cropped<com_2d[2]+cube[2]/2.)
    mask = mask.reshape(-1)
    x = x.reshape(-1)[mask]
    y = y.reshape(-1)[mask]
    z = z.reshape(-1)[mask]

    # Normalize x, y and z to [0, 1)
    x = x/H
    y = y/W
    z = (z-com_2d[2]+cube[2]/2)/cube[2]

    # Get voxel
    voxel = np.zeros([voxel_len, voxel_len, voxel_len], dtype=np.int)
    x = (x*voxel_len).astype(np.int)
    y = (y*voxel_len).astype(np.int)
    z = (z*voxel_len).astype(np.int)
    voxel[x, y, z] = 1

    return voxel
