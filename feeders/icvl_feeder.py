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
from utils.point_transform import transform_2D_to_3D

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s %(name)s:%(lineno)d] %(message)s")
logger = logging.getLogger(__file__)


class ICVLFeeder(Dataset):
    def __init__(self, phase='train', max_jitter=10., depth_sigma=1.):
        """

        :param phase: train or test
        :param max_jitter:
        :param depth_sigma:
        """
        self.phase = phase
        self.max_jitter = max_jitter
        self.depth_sigma = depth_sigma
        config_file = os.path.join(root, "config", "dataset", "icvl.json")
        self.config = json.load(open(config_file, 'r'))
        self.fx = self.config['camera']['fx']
        self.fy = self.config['camera']['fy']
        self.u0 = self.config['camera']['u0']
        self.v0 = self.config['camera']['v0']
        self.crop_size = self.config['crop_size']
        self.inter_matrix = np.array([[self.fx, 0, self.u0],
                                      [0, self.fy, self.v0],
                                      [0, 0, 1]], dtype=np.float32)
        self.cube = np.array(self.config["cube"], dtype=np.float32)
        self.joint_2d, self.joint_3d, self.depth_path = self.load_annotation()
        self.index = np.arange(len(self.depth_path))
        logger.info("{} num: {}".format(phase, len(self.index)))

    def load_annotation(self):
        if self.phase == 'train':
            label_path = [os.path.join(self.config['path'], 'Training', 'labels.txt')]
            depth_dir = os.path.join(self.config['path'], 'Training', 'Depth')
        else:
            label_path = [os.path.join(self.config['path'], 'Testing', 'test_seq_1.txt'),
                      os.path.join(self.config['path'], 'Testing', 'test_seq_2.txt')]
            depth_dir = os.path.join(self.config['path'], 'Testing', 'Depth')

        joint_2d_list = []
        depth_path_list = []
        for path in label_path:
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    sp = line.split()
                    depth_path = os.path.join(depth_dir, sp[0])
                    joint_2d = np.array(list(map(float, sp[1:])), np.float32)
                    joint_2d = joint_2d.reshape((-1, 3))
                    depth_path_list.append(depth_path)
                    joint_2d_list.append(joint_2d)
        joint_2d = np.stack(joint_2d_list, axis=0)
        joint_3d = transform_2D_to_3D(joint_2d, self.fx, self.fy, self.u0, self.v0)
        return joint_2d, joint_3d, depth_path_list

    def __getitem__(self, item):
        item = self.index[item]
        joint_2d, joint_3d, depth_path = self.joint_2d[item], self.joint_3d[item], self.depth_path[item]

        try:
            depth = np.asarray(Image.open(depth_path), np.float32)
        except FileNotFoundError:
            return item, None, None, joint_3d, None, None, self.inter_matrix

        com_3d = np.mean(joint_3d[[0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15]], axis=0)

        if self.max_jitter>0.:
            com_offset = np.random.uniform(low=-1., high=1., size=(3,))*self.max_jitter
            com_3d = com_3d + com_offset
        com_2d = self.inter_matrix @ com_3d[:, None]
        com_2d = np.squeeze(com_2d)
        com_2d[:2] /= com_2d[2]
        com_2d = com_2d.astype(np.float32)

        cube = self.cube
        try:
            cropped, crop_trans, com_2d = crop_area_3d(depth, com_2d, self.fx, self.fy, size=cube,
                                                   dsize=[self.crop_size, self.crop_size], docom=False)
        except UserWarning:
            return item, None, None, joint_3d, None, None, self.inter_matrix
        # plt.imshow(depth)
        # plt.scatter(com_2d[0], com_2d[1])
        # plt.show()
        # plt.imshow(cropped)
        # plt.show()

        if self.depth_sigma>0.:
            # noise = np.random.randn(self.crop_size, self.crop_size)*self.noise_sigma
            noise = np.random.normal(0, self.depth_sigma, size=(self.crop_size, self.crop_size)).astype(np.float32)
            cropped[cropped>1e-3] += noise[cropped>1e-3]

        return item, depth[None, ...], cropped[None, ...], joint_3d, np.array(crop_trans), com_2d, self.inter_matrix, \
               cube

    def __len__(self):
        return len(self.index)


if __name__ == '__main__':
    from tqdm import tqdm
    from feeders.nyu_feeder import collate_fn
    train_dataset = ICVLFeeder('train', max_jitter=0., depth_sigma=0.)
    dataloader = DataLoader(train_dataset, shuffle=False, batch_size=4, collate_fn=collate_fn, num_workers=4)
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = batch_data

    # print(item)
    # print(depth.shape)
    # print(cropped.shape)
    # print(joint_3d.shape)
    # print(crop_trans.shape)
    # print(com_2d.shape)
    # print(inter_matrix.shape)
    # print(cube.shape)
