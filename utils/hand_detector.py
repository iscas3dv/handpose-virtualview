"""
Copyright 2015, 2018 ICG, Graz University of Technology
This file is part of PreView.
PreView is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
PreView is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with PreView.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cv2
import math
from scipy import ndimage
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__file__)


def normlize_depth(depth, com_2d, cube_z):
    norm_depth = depth.copy()
    norm_depth[norm_depth==0] = com_2d[2]+(cube_z/2.)
    norm_depth -= com_2d[2]
    norm_depth /= (cube_z/2.)
    return norm_depth


def calculate_com_2d(dpt):
    """Calculate the center of mass

    :param dpt: depth image; invalid pixels which should not be considered must be set zero
    :return: (x,y,z) center of mass
    """
    dc = dpt.copy()
    cc = ndimage.measurements.center_of_mass(dc > 0)
    num = np.count_nonzero(dc)
    com_2d = np.array((cc[1] * num, cc[0] * num, dc.sum()), np.float32)

    if num == 0:
        return np.array((0, 0, 0), np.float32)
    else:
        return com_2d / num


def calc_mask(depth, com_2d, fx, fy, bbx, offset, minRatioInside=0.75, size=(250, 250, 250)):
    if len(size) != 3:
        raise ValueError("Size must be 3D and dsize 2D bounding box")

    if bbx is not None:
        if len(bbx)==6:
            left, right, up, down, front, back = bbx
        else:
            left, right, up, down = bbx
        left = int(math.floor(left * com_2d[2] / fx - offset) / com_2d[2] * fx)
        right = int(math.floor(right * com_2d[2] / fx + offset) / com_2d[2] * fx)
        up = int(math.floor(up * com_2d[2] / fx - offset) / com_2d[2] * fx)
        down = int(math.floor(down * com_2d[2] / fx + offset) / com_2d[2] * fx)
        left = max(left, 0)
        right = min(right, depth.shape[1])
        up = max(up, 0)
        down = min(down, depth.shape[0])
        imgDepth = np.zeros_like(depth)
        imgDepth[up:down, left:right] = depth[up:down, left:right]
        if len(bbx)==6:
            imgDepth[imgDepth < front-offset] = 0.
            imgDepth[imgDepth > back+offset] = 0.
    else:
        imgDepth = depth

    # calculate boundaries
    zstart = com_2d[2] - size[2] / 2.
    zend = com_2d[2] + size[2] / 2.
    xstart = int(math.floor((com_2d[0] * com_2d[2] / fx - size[0] / 2.) / com_2d[2] * fx))
    xend = int(math.floor((com_2d[0] * com_2d[2] / fx + size[0] / 2.) / com_2d[2] * fx))
    ystart = int(math.floor((com_2d[1] * com_2d[2] / fy - size[1] / 2.) / com_2d[2] * fy))
    yend = int(math.floor((com_2d[1] * com_2d[2] / fy + size[1] / 2.) / com_2d[2] * fy))

    # Check if part within image is large enough; otherwise stop
    xstartin = max(xstart, 0)
    xendin = min(xend, imgDepth.shape[1])
    ystartin = max(ystart, 0)
    yendin = min(yend, imgDepth.shape[0])
    ratioInside = float((xendin - xstartin) * (yendin - ystartin)) / float((xend - xstart) * (yend - ystart))
    if (ratioInside < minRatioInside) and (
            (com_2d[0] < 0) or (com_2d[0] >= imgDepth.shape[1]) or (com_2d[1] < 0) or (
            com_2d[1] >= imgDepth.shape[0])):
        # print("Hand largely outside image (ratio (inside) = {})".format(ratioInside))
        raise UserWarning('Hand not inside image')

    if (ystartin<yendin) and (xstartin<xendin):
        mask = np.zeros_like(imgDepth, dtype=np.int)
        depth_tmp = imgDepth[ystartin:yendin, xstartin:xendin]
        msk = np.bitwise_and(zstart<depth_tmp, depth_tmp<zend)
        mask[ystartin:yendin, xstartin:xendin][msk] = 1
    else:
        raise UserWarning("No hand.")

    # Sanity check
    numValidPixels = np.sum(mask != 0)
    if (numValidPixels < 40):
        # plt.imshow(rz)
        # plt.show()
        # print("Too small number of foreground/hand pixels (={})".format(numValidPixels))
        raise UserWarning("No valid hand. Foreground region too small.")

    return mask


def crop_area_3d(depth, com_2d, fx, fy, bbx=None, offset=0., minRatioInside=0.75,
               size=(250, 250, 250), dsize=(128, 128), docom=True):
    """
    Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
    :param com_2d: center of mass, in image coordinates (x,y,z), z in mm
    :param size: (x,y,z) extent of the source crop volume in mm
    :param dsize: (x,y) extent of the destination size
    :param docom: whether calcate center of mass on cropped
    :return: cropped hand image, transformation matrix for joints
    """
    CROP_BG_VALUE = 0.0

    if len(size) != 3 or len(dsize) != 2:
        # print('Size must be 3D and dsize 2D bounding box')
        raise ValueError("Size must be 3D and dsize 2D bounding box")

    if bbx is not None:
        if len(bbx)==6:
            left, right, up, down, front, back = bbx
        else:
            left, right, up, down = bbx
        left = int(math.floor(left * com_2d[2] / fx - offset) / com_2d[2] * fx)
        right = int(math.floor(right * com_2d[2] / fx + offset) / com_2d[2] * fx)
        up = int(math.floor(up * com_2d[2] / fx - offset) / com_2d[2] * fx)
        down = int(math.floor(down * com_2d[2] / fx + offset) / com_2d[2] * fx)
        left = max(left, 0)
        right = min(right, depth.shape[1])
        up = max(up, 0)
        down = min(down, depth.shape[0])
        imgDepth = np.zeros_like(depth)
        imgDepth[up:down, left:right] = depth[up:down, left:right]
        if len(bbx)==6:
            imgDepth[imgDepth < front-offset] = 0.
            imgDepth[imgDepth > back+offset] = 0.
    else:
        imgDepth = depth

    # calculate boundaries
    zstart = com_2d[2] - size[2] / 2.
    zend = com_2d[2] + size[2] / 2.
    xstart = int(math.floor((com_2d[0] * com_2d[2] / fx - size[0] / 2.) / com_2d[2] * fx))
    xend = int(math.floor((com_2d[0] * com_2d[2] / fx + size[0] / 2.) / com_2d[2] * fx))
    ystart = int(math.floor((com_2d[1] * com_2d[2] / fy - size[1] / 2.) / com_2d[2] * fy))
    yend = int(math.floor((com_2d[1] * com_2d[2] / fy + size[1] / 2.) / com_2d[2] * fy))

    # Check if part within image is large enough; otherwise stop
    xstartin = max(xstart, 0)
    xendin = min(xend, imgDepth.shape[1])
    ystartin = max(ystart, 0)
    yendin = min(yend, imgDepth.shape[0])
    ratioInside = float((xendin - xstartin) * (yendin - ystartin)) / float((xend - xstart) * (yend - ystart))
    if (ratioInside < minRatioInside) and (
            (com_2d[0] < 0) or (com_2d[0] >= imgDepth.shape[1]) or (com_2d[1] < 0) or (com_2d[1] >= imgDepth.shape[0])):
        # print("Hand largely outside image (ratio (inside) = {})".format(ratioInside))
        raise UserWarning('Hand not inside image')

    # crop patch from source
    cropped = imgDepth[max(ystart, 0):min(yend, imgDepth.shape[0]), max(xstart, 0):min(xend, imgDepth.shape[1])].copy()
    # add pixels that are out of the image in order to keep aspect ratio
    cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0), abs(yend) - min(yend, imgDepth.shape[0])),
                                  (abs(xstart) - max(xstart, 0), abs(xend) - min(xend, imgDepth.shape[1]))),
                        mode='constant', constant_values=int(CROP_BG_VALUE))
    msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
    msk2 = np.bitwise_and(cropped > zend, cropped != 0)
    cropped[msk1] = CROP_BG_VALUE  # backface is at 0, it is set later; setting anything outside cube to same value now (was set to zstart earlier)
    cropped[msk2] = CROP_BG_VALUE  # backface is at 0, it is set later

    # for simulating COM within cube
    if docom is True:
        com_2d = calculate_com_2d(cropped)
        if np.allclose(com_2d, 0.):
            com_2d[2] = cropped[cropped.shape[0] // 2, cropped.shape[1] // 2]
        com_2d[0] += xstart
        com_2d[1] += ystart

        # calculate boundaries
        zstart = com_2d[2] - size[2] / 2.
        zend = com_2d[2] + size[2] / 2.
        xstart = int(math.floor((com_2d[0] * com_2d[2] / fx - size[0] / 2.) / com_2d[2] * fx))
        xend = int(math.floor((com_2d[0] * com_2d[2] / fx + size[0] / 2.) / com_2d[2] * fx))
        ystart = int(math.floor((com_2d[1] * com_2d[2] / fy - size[1] / 2.) / com_2d[2] * fy))
        yend = int(math.floor((com_2d[1] * com_2d[2] / fy + size[1] / 2.) / com_2d[2] * fy))

        # crop patch from source
        cropped = imgDepth[max(ystart, 0):min(yend, imgDepth.shape[0]),
                  max(xstart, 0):min(xend, imgDepth.shape[1])].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0), abs(yend) - min(yend, imgDepth.shape[0])),
                                      (abs(xstart) - max(xstart, 0), abs(xend) - min(xend, imgDepth.shape[1]))),
                            mode='constant', constant_values=0)
        msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
        msk2 = np.bitwise_and(cropped > zend, cropped != 0)
        cropped[msk1] = zstart
        cropped[msk2] = CROP_BG_VALUE  # backface is at 0, it is set later

    wb = (xend - xstart)
    hb = (yend - ystart)
    trans = np.asmatrix(np.eye(3, dtype=np.float32))
    trans[0, 2] = -xstart
    trans[1, 2] = -ystart
    # compute size of image patch for isotropic scaling where the larger side is the side length of the fixed size image patch (preserving aspect ratio)
    if wb > hb:
        sz = (dsize[0], int(round(hb * dsize[0] / float(wb))))
    else:
        sz = (int(round(wb * dsize[1] / float(hb))), dsize[1])

    # comdpute scale factor from cropped ROI in image to fixed size image patch; set up matrix with same scale in x and y (preserving aspect ratio)
    roi = cropped
    if roi.shape[0] > roi.shape[1]:  # Note, roi.shape is (y,x) and sz is (x,y)
        scale = np.asmatrix(np.eye(3, dtype=np.float32) * sz[1] / float(roi.shape[0]))
    else:
        scale = np.asmatrix(np.eye(3, dtype=np.float32) * sz[0] / float(roi.shape[1]))
    scale[2, 2] = 1

    # depth resize
    rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)

    # Sanity check
    numValidPixels = np.sum(rz != CROP_BG_VALUE)
    if (numValidPixels < 40) or (numValidPixels < (np.prod(dsize) * 0.01)):
        # plt.imshow(rz)
        # plt.show()
        # print("Too small number of foreground/hand pixels (={})".format(numValidPixels))
        raise UserWarning("No valid hand. Foreground region too small.")

    # Place the resized patch (with preserved aspect ratio) in the center of a fixed size patch (padded with default background values)
    ret = np.ones(dsize, np.float32) * CROP_BG_VALUE  # use background as filler
    xstart = int(math.floor(dsize[0] / 2 - rz.shape[1] / 2))
    xend = int(xstart + rz.shape[1])
    ystart = int(math.floor(dsize[1] / 2 - rz.shape[0] / 2))
    yend = int(ystart + rz.shape[0])
    ret[ystart:yend, xstart:xend] = rz
    # print rz.shape
    off = np.asmatrix(np.eye(3, dtype=np.float32))
    off[0, 2] = xstart
    off[1, 2] = ystart

    # Transformation from original image to fixed size crop includes
    # the translation of the "anchor" point of the crop to origin (=trans),
    # the (isotropic) scale factor (=scale), and
    # the offset of the patch (with preserved aspect ratio) within the fixed size patch (=off)
    return ret, off * scale * trans, com_2d
