import os
import sys
root = os.path.dirname(os.path.abspath(__file__))
import shutil
from utils.parser_utils import get_view_a2j_parser
from ops.image_ops import normalize_image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from feeders.nyu_feeder import NyuFeeder, collate_fn
from feeders.icvl_feeder import ICVLFeeder
from torch.utils.data.dataloader import DataLoader
import json
import argparse
from models.multiview_a2j import MultiviewA2J
from models.conf_net import ConfNet
from models.view_selector_a2j import ViewSelector
from ops.loss_ops import ViewSelectA2JLossCalculator
import numpy as np
import random
import yaml
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from ops.point_transform import transform_3D_to_2D
import time
from tqdm import tqdm
import cv2
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s %(name)s:%(lineno)d] %(message)s")
logger = logging.getLogger(__file__)


def init_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Processor(object):
    def __init__(self, args):
        self.args = args
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(list(map(str, self.args.gpus)))
        logger.info("CUDA_VISIBLE_DEVICES: " + os.environ["CUDA_VISIBLE_DEVICES"])
        self.dataset_config = json.load(open("config/dataset/{}.json".format(args.dataset), 'r'))
        self.num_joints = len(self.dataset_config['selected'])
        self.fx = self.dataset_config['camera']['fx']
        self.fy = self.dataset_config['camera']['fy']
        self.u0 = self.dataset_config['camera']['u0']
        self.v0 = self.dataset_config['camera']['v0']
        if args.level==1:
            self.num_views = 3
        elif args.level==2:
            self.num_views = 9
        elif args.level==3:
            self.num_views = 15
        elif args.level==4:
            self.num_views = 25
        elif args.level==5:
            self.num_views = 81

        if self.args.phase == 'train':
            self.train_log_dir = os.path.join(self.args.log_dir, 'train')
            self.test_log_dir = os.path.join(self.args.log_dir, 'test')
            if not self.args.resume_training:
                for d in [self.args.log_dir, self.train_log_dir, self.test_log_dir]:
                    if os.path.exists(d):
                        shutil.rmtree(d)
            for d in [self.args.log_dir, self.train_log_dir, self.test_log_dir, self.args.model_saved_path]:
                os.makedirs(d, exist_ok=True)
            handler = logging.FileHandler(
                filename=os.path.join(args.log_dir, 'train_log.txt'),
                mode='a' if self.args.resume_training else 'w')
        elif self.args.phase == 'eval':
            if os.path.exists(self.args.log_dir):
                shutil.rmtree(self.args.log_dir)
            os.makedirs(self.args.log_dir, exist_ok=True)
            handler = logging.FileHandler(
                filename=os.path.join(args.log_dir, 'eval_log.txt'),
                mode='a' if self.args.resume_training else 'w')

        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s: %(levelname)s %(name)s:%(lineno)d] %(message)s")
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

        self.global_step = self.start_epoch = 0
        self.model_saved_name = os.path.join(self.args.model_saved_path, "model.pth")
        self.save_args()
        self.dataset, self.feeder = self.load_data()
        self.view_selector, self.conf_net, self.loss_calc = self.load_model()
        if self.args.phase == 'train':
            self.optimizer = self.load_optimizer()
            self.scheduler = ExponentialLR(self.optimizer, gamma=self.args.learning_decay_rate)
            self.min_error_3d = 1e9
            if self.args.resume_training:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(self.checkpoint['scheduler'])
                self.global_step = self.checkpoint['global_step']
                self.start_epoch = self.checkpoint['epoch']
                self.min_error_3d = self.checkpoint['error_3d']
            self.train_writer = SummaryWriter(self.train_log_dir)
            self.test_writer = SummaryWriter(self.test_log_dir)

    def save_args(self):
        arg_dict = vars(self.args)
        with open('{}/{}_config.yaml'.format(self.args.log_dir, self.args.phase), 'w') as f:
            yaml.dump(arg_dict, f)

    def load_data(self):
        feeder = dict()
        dataset = dict()
        if self.args.phase == 'train':
            if self.args.dataset == 'nyu':
                train_set = NyuFeeder('train', max_jitter=self.args.max_jitter, depth_sigma=self.args.depth_sigma,
                                      offset=self.args.offset, random_flip=self.args.random_flip,
                                      adjust_cube=self.args.adjust_cube)
            elif self.args.dataset == 'icvl':
                train_set = ICVLFeeder('train', max_jitter=self.args.max_jitter, depth_sigma=self.args.depth_sigma)
            dataset['train'] = train_set
            lengths = [len(train_set)//self.args.split] * (self.args.split-1)
            lengths.append(len(train_set)-sum(lengths))
            train_set_split = torch.utils.data.random_split(train_set, lengths)
            feeder['train'] = [DataLoader(
                dataset=train_set_split[i],
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_worker,
                drop_last=True,
                collate_fn=collate_fn
            ) for i in range(self.args.split)]
        if self.args.dataset == 'nyu':
            test_set = NyuFeeder('test', max_jitter=0., depth_sigma=0., offset=self.args.offset, random_flip=False,
                                 adjust_cube=self.args.adjust_cube)
        elif self.args.dataset == 'icvl':
            test_set = ICVLFeeder('test', max_jitter=0., depth_sigma=0.)
        dataset['test'] = test_set
        feeder['test'] = DataLoader(
            dataset=test_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_worker,
            drop_last=False,
            collate_fn=collate_fn
        )
        return dataset, feeder

    def load_model(self):
        multiview_a2j = MultiviewA2J(self.dataset_config['camera'], self.num_joints, self.args.n_head, self.args.d_attn,
                             self.args.d_k, self.args.d_v, self.args.d_inner, 0., self.args.num_select)
        self.a2j_checkpoint = torch.load(self.args.pre_a2j)
        multiview_a2j.load_state_dict(self.a2j_checkpoint['model_state_dict'])

        conf_net = ConfNet(self.num_views, self.args.dropout_rate)

        view_selector = ViewSelector(multiview_a2j, conf_net, self.args.random_select)
        if self.args.pre_model_path is not None:
            self.checkpoint = torch.load(self.args.pre_model_path)
            conf_net.load_state_dict(self.checkpoint['model_state_dict'])
        loss_calc = ViewSelectA2JLossCalculator(self.args.alpha, self.args.conf_factor)
        view_selector = nn.DataParallel(view_selector).cuda()
        loss_calc = nn.DataParallel(loss_calc).cuda()
        return view_selector, conf_net, loss_calc

    def load_optimizer(self):
        optimizer_parameters = []
        for param in self.view_selector.named_parameters():
            if 'conf_net' in param[0]:
                optimizer_parameters.append(param[1])
            else:
                param[1].requires_grad = False
        optimizer = optim.Adam(
            optimizer_parameters,
            lr=self.args.learning_rate,
            weight_decay=self.args.reg_weight
        )
        return optimizer

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch):
        self.view_selector.train()
        logger.info('Train epoch: {}'.format(epoch + 1))
        timer = {'dataloader': 0., 'model': 0., 'statistics': 0.}
        if epoch%(self.args.split-1) == 0:
            np.random.shuffle(self.dataset['train'].index)
        feeder = self.feeder['train'][epoch%(self.args.split-1)]
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        epoch_conf_loss = epoch_loss = epoch_error_3d_fused_select_light = \
        epoch_error_3d_conf_select_light = epoch_error_3d_fused = epoch_error_3d_conf_select = 0.
        num_sample = 0
        for batch_idx, data in enumerate(tqdm(feeder, ncols=80)):
            item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = data
            timer['dataloader'] += self.split_time()
            cropped = cropped.cuda()
            joint_3d = joint_3d.cuda()
            crop_trans = crop_trans.cuda()
            com_2d = com_2d.cuda()
            inter_matrix = inter_matrix.cuda()
            cube = cube.cuda()
            crop_expand, view_trans, anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm, \
            joints_3d_pred, joint_3d_fused, conf, joint_3d_conf_select, joints_3d_pred_select_light, \
            joint_3d_fused_select_light, joint_3d_conf_select_light, conf_light = \
                self.view_selector(cropped, crop_trans, com_2d, inter_matrix, cube, self.args.level,
                                   self.args.num_select, inference=False)
            conf_loss, loss, error_3d_fused_select_light, error_3d_conf_select_light,\
            error_3d_fused, error_3d_conf_select = self.loss_calc(
                joints_3d_pred, joint_3d_fused, conf, joint_3d_conf_select,
                joints_3d_pred_select_light, joint_3d_fused_select_light, joint_3d_conf_select_light, conf_light,
                view_trans, crop_trans, com_2d, cube, self.fx, self.fy, self.u0, self.v0, joint_3d)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            epoch_conf_loss += torch.sum(conf_loss).item()
            epoch_loss += torch.sum(loss).item()
            epoch_error_3d_fused_select_light += torch.sum(error_3d_fused_select_light).item()
            epoch_error_3d_conf_select_light += torch.sum(error_3d_conf_select_light).item()
            epoch_error_3d_fused += torch.sum(error_3d_fused).item()
            epoch_error_3d_conf_select += torch.sum(error_3d_conf_select).item()
            num_sample += cropped.size(0)
            timer['model'] += self.split_time()

            self.train_writer.add_scalar('conf_loss', conf_loss.mean(), global_step=self.global_step)
            self.train_writer.add_scalar('loss', loss.mean(), global_step=self.global_step)
            self.train_writer.add_scalar('error_3d_fused_select_light', error_3d_fused_select_light.mean(),
                                         global_step=self.global_step)
            self.train_writer.add_scalar('error_3d_conf_select_light', error_3d_conf_select_light.mean(),
                                         global_step=self.global_step)
            self.train_writer.add_scalar('error_3d_fused', error_3d_fused.mean(),
                                         global_step=self.global_step)
            self.train_writer.add_scalar('error_3d_conf_select', error_3d_conf_select.mean(),
                                         global_step=self.global_step)
            if self.global_step%100==0:
                self.train_writer.add_images('crop_expand', (crop_expand[0][:] + 1.) / 2.,
                                             global_step=self.global_step)
                conf_show = conf[:4]
                conf_show = conf_show / conf_show.max(dim=1)[0][:, None]
                self.train_writer.add_images('conf', conf_show.reshape((4, 1, 5, 5)),
                                            global_step=self.global_step)
                conf_light_show = conf_light[:4]
                conf_light_show = conf_light_show / conf_light_show.max(dim=1)[0][:, None]
                self.train_writer.add_images('conf_light', conf_light_show.reshape((4, 1, 5, 5)),
                                            global_step=self.global_step)

            self.global_step += 1
            timer['statistics'] += self.split_time()

        epoch_conf_loss /= num_sample
        epoch_loss /= num_sample
        epoch_error_3d_fused_select_light /= num_sample
        epoch_error_3d_conf_select_light /= num_sample
        epoch_error_3d_fused /= num_sample
        epoch_error_3d_conf_select /= num_sample
        self.scheduler.step()
        timer['model'] += self.split_time()

        lr = self.optimizer.param_groups[0]['lr']
        self.train_writer.add_scalar('lr', lr, self.global_step)
        timer['statistics'] += self.split_time()

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        logger.info('Mean training epoch_conf_loss: {:.6f}'.format(epoch_conf_loss))
        logger.info('Mean training epoch_loss: {:.6f}'.format(epoch_loss))
        logger.info('Mean training epoch_error_3d_fused_select_light: {:.2f}mm'.format(epoch_error_3d_fused_select_light))
        logger.info('Mean training epoch_error_3d_conf_select_light: {:.2f}mm'.format(epoch_error_3d_conf_select_light))
        logger.info('Mean training epoch_error_3d_fused: {:.2f}mm'.format(epoch_error_3d_fused))
        logger.info('Mean training epoch_error_3d_conf_select: {:.2f}mm'.format(epoch_error_3d_conf_select))
        logger.info('Time consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

    def eval(self, epoch):
        self.view_selector.eval()
        logger.info('Test epoch: {}'.format(epoch + 1))
        timer = {'dataloader': 0., 'model': 0., 'statistics': 0.}
        feeder = self.feeder['test']
        self.record_time()
        epoch_conf_loss = epoch_loss = epoch_error_3d_fused_select_light = \
        epoch_error_3d_conf_select_light = epoch_error_3d_fused = epoch_error_3d_conf_select = 0.
        num_sample = 0
        if self.args.save_result:
            joint_3d_list = []
            joint_2d_list = []
        for batch_idx, data in enumerate(tqdm(feeder, ncols=80)):
            with torch.no_grad():
                item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = data
                timer['dataloader'] += self.split_time()
                cropped = cropped.cuda()
                joint_3d = joint_3d.cuda()
                crop_trans = crop_trans.cuda()
                com_2d = com_2d.cuda()
                inter_matrix = inter_matrix.cuda()
                cube = cube.cuda()
                crop_expand, view_trans, anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm, \
                joints_3d_pred, joint_3d_fused, conf, joint_3d_conf_select, joints_3d_pred_select_light, \
                joint_3d_fused_select_light, joint_3d_conf_select_light, conf_light = \
                    self.view_selector(cropped, crop_trans, com_2d, inter_matrix, cube, self.args.level,
                                       self.args.num_select, inference=False)
                conf_loss, loss, error_3d_fused_select_light, error_3d_conf_select_light, \
                error_3d_fused, error_3d_conf_select = self.loss_calc(
                    joints_3d_pred, joint_3d_fused, conf, joint_3d_conf_select,
                    joints_3d_pred_select_light, joint_3d_fused_select_light, joint_3d_conf_select_light, conf_light,
                    view_trans, crop_trans, com_2d, cube, self.fx, self.fy, self.u0, self.v0, joint_3d)
                epoch_conf_loss += torch.sum(conf_loss).item()
                epoch_loss += torch.sum(loss).item()
                epoch_error_3d_fused_select_light += torch.sum(error_3d_fused_select_light).item()
                epoch_error_3d_conf_select_light += torch.sum(error_3d_conf_select_light).item()
                epoch_error_3d_fused += torch.sum(error_3d_fused).item()
                epoch_error_3d_conf_select += torch.sum(error_3d_conf_select).item()
                num_sample += cropped.size(0)
                if self.args.save_result:
                    joint_3d_list.append(joint_3d_conf_select.cpu().numpy())
                    joint_2d_pred = transform_3D_to_2D(joint_3d_conf_select, self.fx, self.fy, self.u0, self.v0)
                    # print(joint_2d_pred)
                    joint_2d_list.append(joint_2d_pred.cpu().numpy())
                timer['model'] += self.split_time()
                if self.args.phase == 'train' and batch_idx % 100 == 0:
                    self.test_writer.add_images('crop_expand', (crop_expand[0][:] + 1.) / 2.,
                                                 global_step=self.global_step+batch_idx)
                    conf_show = conf[:4]
                    conf_show = conf_show / conf_show.max(dim=1)[0][:, None]
                    self.test_writer.add_images('conf', conf_show.reshape((4, 1, 5, 5)),
                                                global_step=self.global_step+batch_idx)
                    conf_light_show = conf_light[:4]
                    conf_light_show = conf_light_show / conf_light_show.max(dim=1)[0][:, None]
                    self.test_writer.add_images('conf_light', conf_light_show.reshape((4, 1, 5, 5)),
                                                 global_step=self.global_step+batch_idx)

        epoch_conf_loss /= num_sample
        epoch_loss /= num_sample
        epoch_error_3d_fused_select_light /= num_sample
        epoch_error_3d_conf_select_light /= num_sample
        epoch_error_3d_fused /= num_sample
        epoch_error_3d_conf_select /= num_sample
        timer['model'] += self.split_time()

        if self.args.phase == 'train':
            self.test_writer.add_scalar('conf_loss', conf_loss.mean(), global_step=self.global_step)
            self.test_writer.add_scalar('loss', loss.mean(), global_step=self.global_step)
            self.test_writer.add_scalar('error_3d_fused_select_light', epoch_error_3d_fused_select_light,
                                        global_step=self.global_step)
            self.test_writer.add_scalar('error_3d_conf_select_light', epoch_error_3d_conf_select_light,
                                        global_step=self.global_step)
            self.test_writer.add_scalar('error_3d_fused', epoch_error_3d_fused,
                                         global_step=self.global_step)
            self.test_writer.add_scalar('error_3d_conf_select', epoch_error_3d_conf_select,
                                         global_step=self.global_step)
            timer['statistics'] += self.split_time()

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        total_time = timer['dataloader'] + timer['model'] + timer['statistics']

        logger.info('Mean test epoch_conf_loss: {:.6f}'.format(epoch_conf_loss))
        logger.info('Mean test epoch_loss: {:.6f}'.format(epoch_loss))
        logger.info('Mean test epoch_error_3d_fused_select_light: {:.2f}mm'.format(epoch_error_3d_fused_select_light))
        logger.info('Mean test epoch_error_3d_conf_select_light: {:.2f}mm'.format(epoch_error_3d_conf_select_light))
        logger.info('Mean test epoch_error_3d_fused: {:.2f}mm'.format(epoch_error_3d_fused))
        logger.info('Mean test epoch_error_3d_conf_select: {:.2f}mm'.format(epoch_error_3d_conf_select))
        logger.info('Time consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        logger.info('FPS: {:.2f}'.format(num_sample / total_time))

        if self.args.save_result:
            joint_3d_pred = np.concatenate(joint_3d_list, axis=0)
            if self.args.dataset == 'nyu':
                joint_3d_pred[:, :, 1] = -joint_3d_pred[:, :, 1]
            joint_3d_pred = joint_3d_pred.reshape(joint_3d_pred.shape[0], -1)
            joint_2d_pred = np.concatenate(joint_2d_list, axis=0)
            joint_2d_pred = joint_2d_pred.reshape(joint_2d_pred.shape[0], -1)

        if self.args.phase == 'train' and epoch_error_3d_conf_select_light < self.min_error_3d:
            self.min_error_3d = epoch_error_3d_conf_select_light
            state = {
                'epoch': epoch+1,
                'global_step': self.global_step,
                'scheduler': self.scheduler.state_dict(),
                'model_state_dict': self.conf_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'error_3d': epoch_error_3d_conf_select_light
            }
            torch.save(state, self.model_saved_name)
            if self.args.save_result:
                np.savetxt(os.path.join(self.args.log_dir, 'joint_3d.txt'), joint_3d_pred, fmt='%.3f')
                np.savetxt(os.path.join(self.args.log_dir, 'joint_2d.txt'), joint_2d_pred, fmt='%.3f')

        if self.args.phase == 'eval' and self.args.save_result:
            np.savetxt(os.path.join(self.args.log_dir, 'joint_3d.txt'), joint_3d_pred, fmt='%.3f')
            np.savetxt(os.path.join(self.args.log_dir, 'joint_2d.txt'), joint_2d_pred, fmt='%.3f')

    def inference(self):
        self.view_selector.eval()
        timer = {'dataloader': 0., 'model': 0., 'statistics': 0.}
        feeder = self.feeder['test']
        self.record_time()
        epoch_error_3d_fused_select = epoch_error_3d_conf_select = 0.
        num_sample = 0
        if self.args.save_result:
            joint_3d_list = []
            joint_2d_list = []
        for batch_idx, data in enumerate(tqdm(feeder, ncols=80)):
            with torch.no_grad():
                item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = data
                timer['dataloader'] += self.split_time()
                cropped = cropped.cuda()
                joint_3d = joint_3d.cuda()
                crop_trans = crop_trans.cuda()
                com_2d = com_2d.cuda()
                inter_matrix = inter_matrix.cuda()
                cube = cube.cuda()
                joints_3d_pred_select, joint_3d_fused_select, joint_3d_conf_select = \
                    self.view_selector(cropped, crop_trans, com_2d, inter_matrix, cube, self.args.level,
                                       self.args.num_select, inference=True)
                error_3d_fused = torch.norm(joint_3d_fused_select - joint_3d, dim=-1).mean(-1)
                error_3d_conf = torch.norm(joint_3d_conf_select - joint_3d, dim=-1).mean(-1)
                epoch_error_3d_fused_select += torch.sum(error_3d_fused).item()
                epoch_error_3d_conf_select += torch.sum(error_3d_conf).item()
                num_sample += cropped.size(0)
                if self.args.save_result:
                    joint_3d_list.append(joint_3d_conf_select.cpu().numpy())
                    joint_2d_pred = transform_3D_to_2D(joint_3d_conf_select, self.fx, self.fy, self.u0, self.v0)
                    # print(joint_2d_pred)
                    joint_2d_list.append(joint_2d_pred.cpu().numpy())
                timer['model'] += self.split_time()

        epoch_error_3d_fused_select /= num_sample
        epoch_error_3d_conf_select /= num_sample
        timer['model'] += self.split_time()

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        total_time = timer['dataloader'] + timer['model'] + timer['statistics']

        logger.info('Mean test epoch_error_3d_fused_select: {:.2f}mm'.format(epoch_error_3d_fused_select))
        logger.info('Mean test epoch_error_3d_conf_select: {:.2f}mm'.format(epoch_error_3d_conf_select))
        logger.info('Time consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        logger.info('FPS: {:.2f}'.format(num_sample / total_time))

        if self.args.save_result:
            joint_3d_pred = np.concatenate(joint_3d_list, axis=0)
            if self.args.dataset == 'nyu':
                joint_3d_pred[:, :, 1] = -joint_3d_pred[:, :, 1]
            joint_3d_pred = joint_3d_pred.reshape(joint_3d_pred.shape[0], -1)
            joint_2d_pred = np.concatenate(joint_2d_list, axis=0)
            joint_2d_pred = joint_2d_pred.reshape(joint_2d_pred.shape[0], -1)

        if self.args.save_result:
            np.savetxt(os.path.join(self.args.log_dir, 'joint_3d.txt'), joint_3d_pred, fmt='%.3f')
            np.savetxt(os.path.join(self.args.log_dir, 'joint_2d.txt'), joint_2d_pred, fmt='%.3f')

    def start(self):
        if self.args.phase == 'train':
            lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', lr, self.global_step)
            for epoch in range(self.start_epoch, self.args.num_epoch):
                self.train(epoch)
                self.eval(epoch)
            logger.info("Min error 3d: {:.2f}mm, model name: {}".format(self.min_error_3d, self.model_saved_name))
        elif self.args.phase == 'eval':
            self.inference()


if __name__ == '__main__':
    parser = get_view_a2j_parser()
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            default_args = yaml.load(f, Loader=yaml.FullLoader)
        keys = vars(args).keys()
        for key in default_args.keys():
            if key not in keys:
                logger.error('Wrong arg: {}'.format(key))
                assert (key in keys)
        parser.set_defaults(**default_args)
        args = parser.parse_args()
    if args.phase == 'train' and args.pre_model_path is None and args.resume_training:
        logger.critical('When parameter "pre_model_path" is None, parameter "resume_training" can not be true.')
        raise ValueError('When parameter "pre_model_path" is None, parameter "resume_training" can not be true.')
    if args.phase == 'train':
        init_seed(args.seed)
    else:
        torch.backends.cudnn.benchmark = True
    processor = Processor(args)
    processor.start()

