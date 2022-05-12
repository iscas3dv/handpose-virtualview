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
from utils.hand_detector import crop_area_3d
from utils.image_utils import normlize_depth
import matplotlib.patches as mpathes
from utils.point_transform import transform_3D_to_2D, transform_2D_to_3D

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s %(name)s:%(lineno)d] %(message)s")
logger = logging.getLogger(__file__)


def get_center_from_bbx(path, img_w, img_h, fx, fy, bbx_rectify=True):
    # Reference: https://github.com/wozhangzhaohui/HandAugment
    cube_len = 150.
    lines = [line.split() for line in open(path).readlines()]
    bb_list = [[int(x) for x in line[1:]] for line in lines]
    center_uvd_list = []
    for bb in bb_list:
        if bb[0]>bb[2] or bb[1]>bb[3]:
            center_uvd_list.append(None)
            continue
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        ww = max(w, h)
        if bbx_rectify:
            if w < ww:
                if bb[0] == 0:
                    bb[0] = bb[2] - ww
                elif bb[2] == img_w:
                    bb[2] = bb[0] + ww
            if h < ww:
                if bb[1] == 0:
                    bb[1] = bb[3] - ww
                elif bb[3] == img_h:
                    bb[3] = bb[1] + ww

        center_uvd = np.array([(bb[0] + bb[2]) / 2,
                               (bb[1] + bb[3]) / 2,
                               cube_len*2 / ww * fx], dtype=np.float32)
        center_uvd_list.append(center_uvd)
    return center_uvd_list


def load_joint_pred(path, fx, fy, u0, v0):
    # Reference: https://github.com/wozhangzhaohui/HandAugment
    joint_3d_list = []
    joint_2d_list = []
    with open(path, 'r') as f:
        for anno in f.readlines():
            anno = anno.split('\t')
            if (anno[-1] == '\n'):
                anno = anno[:-1]
            if len(anno) == 2:
                joint_3d_list.append(None)
                joint_2d_list.append(None)
            else:
                joint_3d = np.array(anno[1:]).astype(np.float32)
                joint_3d = joint_3d.reshape(21, 3)
                joint_2d = transform_3D_to_2D(joint_3d, fx, fy, u0, v0)
                joint_3d_list.append(joint_3d)
                joint_2d_list.append(joint_2d)
    return joint_3d_list, joint_2d_list


class Hands2019Feeder(Dataset):
    def __init__(self, phase='train', max_jitter=10., depth_sigma=0., cube_len=None, min_scale=1., max_scale=1.,
                 offset=30., hand_thickness=20., random_flip=False, use_joint=False):
        self.phase = phase
        self.max_jitter = max_jitter
        self.depth_sigma = depth_sigma
        self.cube_len = cube_len
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.offset = offset
        self.hand_thickness = hand_thickness
        self.random_flip = random_flip
        self.use_joint = use_joint
        config_file = os.path.join(root, "config", "dataset", "hands2019.json")
        self.config = json.load(open(config_file, 'r'))
        self.fx = np.float32(self.config['camera']['fx'])
        self.fy = np.float32(self.config['camera']['fy'])
        self.u0 = np.float32(self.config['camera']['u0'])
        self.v0 = np.float32(self.config['camera']['v0'])
        if cube_len is None:
            self.cube = np.array(self.config['cube'], dtype=np.float32)
        else:
            self.cube = np.array([cube_len, cube_len, cube_len], dtype=np.float32)
        self.crop_size = self.config['crop_size']
        self.inter_matrix = np.array([[self.fx, 0, self.u0],
                                      [0, self.fy, self.v0],
                                      [0, 0, 1]], dtype=np.float32)
        self.depth_name_list, self.joint_3d_list, self.joint_2d_list = self.load_annotation()

        self.com_2d_list = get_center_from_bbx(
            os.path.join(self.config['path'], 'training_bbs.txt'), self.config['width'],self.config['height'],
            self.fx, self.fy)
        if use_joint:
            self.joint_3d_pred_list, self.joint_2d_pred_list = load_joint_pred(
                os.path.join(self.config["path"], 'training_joint.txt'), self.fx, self.fy, self.u0, self.v0)
        num = len(self.depth_name_list)
        test_num = num // 10
        train_num = num - test_num
        if phase == 'train':
            self.depth_name_list = self.depth_name_list[:train_num]
            self.joint_3d_list = self.joint_3d_list[:train_num]
            self.joint_2d_list = self.joint_2d_list[:train_num]
            self.com_2d_list = self.com_2d_list[:train_num]
            if use_joint:
                self.joint_3d_pred_list = self.joint_3d_pred_list[:train_num]
                self.joint_2d_pred_list = self.joint_2d_pred_list[:train_num]
        else:
            self.depth_name_list = self.depth_name_list[train_num:]
            self.joint_3d_list = self.joint_3d_list[train_num:]
            self.joint_2d_list = self.joint_2d_list[train_num:]
            self.com_2d_list = self.com_2d_list[train_num:]
            if use_joint:
                self.joint_3d_pred_list = self.joint_3d_pred_list[train_num:]
                self.joint_2d_pred_list = self.joint_2d_pred_list[train_num:]
        self.index = np.arange(len(self.depth_name_list))

    def load_annotation(self):
        joint_anno_path = os.path.join(self.config["path"], 'training_joint_annotation.txt')
        bbs_path = os.path.join(self.config["path"], 'training_bbs.txt')
        joint_3d_list = []
        joint_2d_list = []
        img_name_list = []
        bbx_list = []
        with open(joint_anno_path, 'r') as f:
            for anno in f.readlines():
                anno = anno.split('\t')
                if (anno[-1] == '\n'):
                    anno = anno[:-1]
                img_name = anno[0]
                joint_3d = np.array(anno[1:]).astype(np.float32)
                joint_3d = joint_3d.reshape(21, 3)
                joint_2d = transform_3D_to_2D(joint_3d, self.fx, self.fy, self.u0, self.v0)
                joint_3d_list.append(joint_3d)
                joint_2d_list.append(joint_2d)
                img_name_list.append(img_name)

        return img_name_list, joint_3d_list, joint_2d_list

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
        depth_path = os.path.join(self.config["path"], 'training_images', self.depth_name_list[item])
        depth = cv2.imread(depth_path, 2).astype(np.float32)
        joint_3d, com_2d = self.joint_3d_list[item], self.com_2d_list[item]
        try:
            if com_2d is None:
                raise ValueError
            if self.max_jitter>0.:
                com_3d = transform_2D_to_3D(com_2d, self.fx, self.fy, self.u0, self.v0)
                com_offset = np.random.uniform(low=-1., high=1., size=(3,))*self.max_jitter
                com_offset = com_offset.astype(np.float32)
                com_3d = com_3d + com_offset
                com_2d = transform_3D_to_2D(com_3d, self.fx, self.fy, self.u0, self.v0)

            scale = np.random.uniform(low=self.min_scale, high=self.max_scale)
            cube = self.cube * scale
            if self.use_joint:
                joint_2d_pred, joint_3d_pred = self.joint_2d_pred_list[item], self.joint_3d_pred_list[item]
                left = np.min(joint_2d_pred[:, 0])
                right = np.max(joint_2d_pred[:, 0])
                up = np.min(joint_2d_pred[:, 1])
                down = np.max(joint_2d_pred[:, 1])
                front = np.min(joint_3d_pred[:, 2])-self.hand_thickness
                back = np.max(joint_3d_pred[:, 2])
                bbx = [left, right, up, down, front, back]
                cropped, crop_trans, com_2d = crop_area_3d(depth, com_2d, self.fx, self.fy, bbx=bbx, offset=self.offset,
                                                           size=cube, dsize=[self.crop_size, self.crop_size], docom=False)
            else:
                cropped, crop_trans, com_2d = crop_area_3d(depth, com_2d, self.fx, self.fy, size=cube,
                                                           dsize=[self.crop_size, self.crop_size], docom=False)
        except Exception as e:
            # exc_type, exc_value, exc_obj = sys.exc_info()
            # traceback.print_tb(exc_obj)
            # print(com_2d)
            # print(self.depth_name_list[item])
            # plt.imshow(depth)
            # plt.show()
            # height = down - up
            # width = right - left
            # rect = mpathes.Rectangle([left, up], width, height, color='r', fill=False, linewidth=2)
            # fig, ax = plt.subplots()
            # ax.imshow(depth)
            # ax.add_patch(rect)
            # plt.show()
            # plt.imshow(mask_)
            # plt.show()
            return item, None, None, None, None, None, None, None

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


def collate_fn(batch):
    batch_item = []
    batch_depth = []
    batch_cropped = []
    batch_joint_3d = []
    batch_crop_trans = []
    batch_com_2d = []
    batch_inter_matrix = []
    batch_cube = []
    for item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube in batch:
        if depth is not None:
            batch_item.append(item)
            batch_depth.append(depth)
            batch_cropped.append(cropped)
            batch_joint_3d.append(joint_3d)
            batch_crop_trans.append(crop_trans)
            batch_com_2d.append(com_2d)
            batch_inter_matrix.append(inter_matrix)
            batch_cube.append(cube)
    output = [torch.from_numpy(np.array(batch_item))]
    for arrays in [batch_depth, batch_cropped, batch_joint_3d, batch_crop_trans, batch_com_2d, batch_inter_matrix,
                   batch_cube]:
        output.append(torch.from_numpy(np.stack(arrays, axis=0)))
    return output


if __name__ == '__main__':
    from tqdm import tqdm
    train_dataset = Hands2019Feeder('train', max_jitter=0., depth_sigma=0., cube_len=270., min_scale=1., max_scale=1.,
                                    offset=30., use_joint=True)
    item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = train_dataset[0]
    plt.imshow(depth[0])
    plt.show()
    print(depth[depth!=0].min())
    print(depth[0, 300, 500:600])
    # dataloader = DataLoader(train_dataset, shuffle=False, batch_size=4, num_workers=1, collate_fn=collate_fn)
    # for batch_idx, batch_data in enumerate(tqdm(dataloader)):
    #     item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = batch_data
    #     break

    # test_dataset = Hands2019Feeder('test', max_jitter=0.)
    # # item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = train_dataset[4979]
    # dataloader = DataLoader(test_dataset, shuffle=False, batch_size=4, num_workers=4, collate_fn=collate_fn)
    # for batch_idx, batch_data in enumerate(tqdm(dataloader)):
    #     item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = batch_data
