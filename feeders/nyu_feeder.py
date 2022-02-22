import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import traceback
from PIL import Image
import scipy.io as sio
import scipy.ndimage
from glob import glob
import json
import logging
import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from utils.hand_detector import calculate_com_2d, crop_area_3d
from utils.image_utils import normlize_depth

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s %(name)s:%(lineno)d] %(message)s")
logger = logging.getLogger(__file__)


class NyuFeeder(Dataset):
    def __init__(self, phase='train', max_jitter=10., depth_sigma=1., offset=20., random_flip=False, adjust_cube=False):
        """

        :param phase: train or test
        :param max_jitter:
        :param depth_sigma:
        :param min_scale:
        :param max_scale:
        :param random_flip:
        :param random_rotate:
        """
        self.phase = phase
        self.max_jitter = max_jitter
        self.depth_sigma = depth_sigma
        self.offset = offset
        self.random_flip = random_flip
        self.adjust_cube = adjust_cube
        config_file = os.path.join(root, "config", "dataset", "nyu.json")
        self.config = json.load(open(config_file, 'r'))
        self.joint_2d, self.joint_3d, self.depth_path = self.load_annotation()
        self.fx = self.config['camera']['fx']
        self.fy = self.config['camera']['fy']
        self.u0 = self.config['camera']['u0']
        self.v0 = self.config['camera']['v0']
        self.crop_size = self.config['crop_size']
        self.inter_matrix = np.array([[self.fx, 0, self.u0],
                                      [0, self.fy, self.v0],
                                      [0, 0, 1]], dtype=np.float32)
        self.cube = np.array(self.config["cube"], dtype=np.float32)
        self.com_2d = [None] * len(self.depth_path)
        # self.index = []
        # if self.phase == 'train':
        #     self.index = [i for i in range(len(self.depth_path)) if i%6!=0]
        # if self.phase == 'test':
        #     self.index = [i for i in range(len(self.depth_path)) if i%6==0]
        self.index = np.arange(len(self.depth_path))
        logger.info("{} num: {}".format(phase, len(self.index)))

    def load_annotation(self):
        data_dir = os.path.join(self.config["path"], self.phase)
        joint_data = sio.loadmat(os.path.join(data_dir, 'joint_data.mat'))
        # if self.phase == 'test':
        #     joint_2d = joint_data['joint_uvd'][0][:, self.config['selected']].astype(np.float32)
        #     joint_3d = joint_data['joint_xyz'][0][:, self.config['selected']].astype(np.float32)
        #     joint_3d[:, :, 1] = -joint_3d[:, :, 1]
        #     depth_path = glob(os.path.join(data_dir, "depth_1_*.png"))
        # else:
        #     joint_2d = joint_data['joint_uvd'][:, :, self.config['selected']].astype(np.float32)
        #     joint_2d = np.reshape(joint_2d, [-1, len(self.config['selected']), 3])
        #     joint_3d = joint_data['joint_xyz'][:, :, self.config['selected']].astype(np.float32)
        #     joint_3d = np.reshape(joint_3d, [-1, len(self.config['selected']), 3])
        #     joint_3d[:, :, 1] = -joint_3d[:, :, 1]
        #     depth_path = glob(os.path.join(data_dir, "depth_*.png"))
        joint_2d = joint_data['joint_uvd'][0][:, self.config['selected']].astype(np.float32)
        joint_3d = joint_data['joint_xyz'][0][:, self.config['selected']].astype(np.float32)
        joint_3d[:, :, 1] = -joint_3d[:, :, 1]
        depth_path = glob(os.path.join(data_dir, "depth_1_*.png"))
        depth_path.sort()
        return joint_2d, joint_3d, depth_path

    def show(self, cropped, joint_3d, crop_trans):
        joint_2d = self.inter_matrix @ np.transpose(joint_3d, (1, 0))
        joint_2d = joint_2d / joint_2d[2, :]
        joint_2d = np.transpose(joint_2d, (1, 0))
        crop_joint_2d = np.ones_like(joint_2d)
        crop_joint_2d[:, :2] = joint_2d[:, :2]
        crop_joint_2d = np.transpose(crop_joint_2d, (1, 0))
        crop_joint_2d = np.array(crop_trans @ crop_joint_2d)
        crop_joint_2d = np.transpose(crop_joint_2d, (1, 0))
        plt.clf()
        plt.imshow(cropped)
        plt.scatter(crop_joint_2d[:, 0], crop_joint_2d[:, 1], c='red')
        plt.show()

    def __getitem__(self, item):
        item = self.index[item]
        joint_2d, joint_3d, depth_path = self.joint_2d[item], self.joint_3d[item], self.depth_path[item]
        depth = load_depth_map(depth_path)
        if depth is None:
            return item, None, None, joint_3d, None, None, self.inter_matrix
        # com_2d = joint_2d[13]
        # com_2d = np.mean(joint_2d, axis=0)
        com_3d = np.mean(joint_3d, axis=0)

        # scale = np.random.uniform(low=self.min_scale, high=self.max_scale)
        # cube = self.cube * scale
        if self.max_jitter>0.:
            com_offset = np.random.uniform(low=-1., high=1., size=(3,))*self.max_jitter
            com_3d = com_3d + com_offset
        com_2d = self.inter_matrix @ com_3d[:, None]
        com_2d = np.squeeze(com_2d)
        com_2d[:2] /= com_2d[2]
        com_2d = com_2d.astype(np.float32)
        if self.adjust_cube:
            distance = np.linalg.norm(joint_3d - com_3d, axis=-1)
            cube_size = (np.max(distance) + self.offset) * 2.
            cube = np.array([cube_size, cube_size, cube_size], dtype=np.float32)
            left = np.min(joint_2d[:, 0])
            right = np.max(joint_2d[:, 0])
            up = np.min(joint_2d[:, 1])
            down = np.max(joint_2d[:, 1])
            front = np.min(joint_3d[:, 2])
            back = np.max(joint_3d[:, 2])
            bbx = [left, right, up, down, front, back]
            cropped, crop_trans, com_2d = crop_area_3d(depth, com_2d, self.fx, self.fy, bbx, self.offset, size=cube,
                                               dsize=(self.crop_size, self.crop_size), docom=False)
        else:
            if self.phase != 'train' and item >= 2440:
                cube = self.cube * 5.0 / 6.0
            else:
                cube = self.cube
            cropped, crop_trans, com_2d = crop_area_3d(depth, com_2d, self.fx, self.fy, size=cube,
                                                       dsize=[self.crop_size, self.crop_size], docom=False)
        # if self.random_rotate:
        #     # plt.imshow(cropped)
        #     # plt.show()
        #     angle = np.random.rand()*360.
        #     M = cv2.getRotationMatrix2D((self.crop_size/2., self.crop_size/2.), angle, 1.)
        #     cropped = cv2.warpAffine(cropped, M, (self.crop_size, self.crop_size), flags=cv2.INTER_NEAREST)
        #     rotate_trans = np.eye(3, dtype=np.float32)
        #     rotate_trans[:2, :] = M
        #     crop_trans = rotate_trans @ crop_trans
        #     # plt.imshow(cropped)
        #     # plt.show()

        if self.random_flip:
            to_center = np.array([[1., 0., self.crop_size/2.],
                                  [0., 1., self.crop_size/2.],
                                  [0., 0., 1]], np.float32)
            to_origin = np.array([[1., 0., -self.crop_size/2.],
                                  [0., 1., -self.crop_size/2.],
                                  [0., 0., 1]], np.float32)
            if random.random()>0.5:
                # Horizontal flip
                cropped = cropped[:, ::-1]
                matrix = np.eye(3, dtype=np.float32)
                matrix[0, 0] = -1
                flip_matrix = to_center @ matrix @ to_origin
                crop_trans = flip_matrix @ crop_trans

            if random.random()>0.5:
                # Vertical flip
                cropped = cropped[::-1, :]
                matrix = np.eye(3, dtype=np.float32)
                matrix[1, 1] = -1
                flip_matrix = to_center @ matrix @ to_origin
                crop_trans = flip_matrix @ crop_trans

            cropped = np.array(cropped)

        if self.depth_sigma>0.:
            # noise = np.random.randn(self.crop_size, self.crop_size)*self.noise_sigma
            noise = np.random.normal(0, self.depth_sigma, size=(self.crop_size, self.crop_size)).astype(np.float32)
            cropped[cropped>1e-3] += noise[cropped>1e-3]

        # self.show(cropped, joint_3d, crop_trans)
        # plt.imshow(depth)
        # plt.show()
        # print(com_2d)
        return item, depth[None, ...], cropped[None, ...], joint_3d, np.array(crop_trans), com_2d, self.inter_matrix, \
               cube

    def __len__(self):
        return len(self.index)


def load_depth_map(filename):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """
    try:
        img = Image.open(filename)
        # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r, np.int32)
        g = np.asarray(g, np.int32)
        b = np.asarray(b, np.int32)
        dpt = np.bitwise_or(np.left_shift(g, 8), b)
        imgdata = np.asarray(dpt, np.float32)
    except IOError as e:
        imgdata = None
        # imgdata = np.zeros((480, 640), np.float32)
        logger.exception(filename+' file broken.')
    return imgdata


def collate_fn(batch):
    # batch_item = []
    # batch_depth = []
    # batch_cropped = []
    # batch_joint_3d = []
    # batch_crop_trans = []
    # batch_com_2d = []
    # batch_inter_matrix = []
    # batch_cube = []
    batch_data = []
    for i in range(len(batch)):
        if batch[i][1] is not None:
            batch_data.append(batch[i])
    # for item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube in batch:
    #     if depth is not None:
    #         batch_item.append(item)
    #         batch_depth.append(depth)
    #         batch_cropped.append(cropped)
    #         batch_joint_3d.append(joint_3d)
    #         batch_crop_trans.append(crop_trans)
    #         batch_com_2d.append(com_2d)
    #         batch_inter_matrix.append(inter_matrix)
    #         batch_cube.append(cube)
    batch_data = list(zip(*batch_data))
    output = [torch.from_numpy(np.array(batch_data[0]))]
    for arrays in batch_data[1:]:
        output.append(torch.from_numpy(np.stack(arrays, axis=0)))
    return output


if __name__ == '__main__':
    train_dataset = NyuFeeder('train', max_jitter=10., depth_sigma=0., offset=30, random_flip=False)
    item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = train_dataset[0]
    dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)
    for batch_idx, batch_data in enumerate(dataloader):
        item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = batch_data
        print(item)
        print(cube)
        break

    # test_dataset = NyuFeeder('test', max_jitter=0., depth_sigma=0., offset=30, random_flip=False)
    # dataloader = DataLoader(test_dataset, shuffle=True, batch_size=4)
    # for batch_idx, batch_data in enumerate(dataloader):
    #     item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = batch_data
    #     print(item)
    #     print(cube)
    #     break

    # random.seed(0)
    # train_dataset = NyuFeeder('test', jitter_sigma=0., noise_sigma=0., scale_sigma=0., random_flip=True)
    # dataloader = DataLoader(train_dataset, shuffle=False, batch_size=4, collate_fn=collate_fn)
    # for batch_idx, batch_data in enumerate(dataloader):
    #     item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = batch_data
    #     print(depth[2, 0, 300, 200])
    #     print(item)
    #     print(cube)
    #     break
