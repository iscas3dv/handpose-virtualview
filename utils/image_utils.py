import numpy as np


def normlize_depth(depth, com_2d, cube_z):
    norm_depth = depth.copy()
    norm_depth[norm_depth==0] = com_2d[2]+(cube_z/2.)
    norm_depth -= com_2d[2]
    norm_depth /= (cube_z/2.)
    return norm_depth


def normlize_image(img):
    """

    :param img: np.array(H, W)
    :return: np.array(H, W)
    """
    t_min = np.min(img)
    t_max = np.max(img)
    img = (img - t_min) / (t_max - t_min)
    return img
