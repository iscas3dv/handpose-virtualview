import os
import shutil
from utils.parser_utils import get_a2j_parser
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from feeders.nyu_feeder import NyuFeeder
from feeders.icvl_feeder import ICVLFeeder
from models.multiview_a2j import MultiviewA2J
from ops.loss_ops import MultiA2JCalculator
from ops.image_ops import normalize_image
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
from feeders.nyu_feeder import collate_fn
from ops.point_transform import transform_3D_to_2D
import time
import json
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s %(name)s:%(lineno)d] %(message)s")
logger = logging.getLogger(__file__)

torch.multiprocessing.set_sharing_strategy('file_system')


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
            self.shape = [1, 3]
        elif args.level==2:
            self.num_views = 9
            self.shape = [3, 3]
        elif args.level==3:
            self.num_views = 15
            self.shape = [3, 5]
        elif args.level==4:
            self.num_views = 25
            self.shape = [5, 5]
        elif args.level==5:
            self.num_views = 81
            self.shape = [9, 9]

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
        self.model, self.loss_calc = self.load_model()
        self.min_error_3d = 1e9
        self.global_step = self.start_epoch = 0
        if self.args.phase == 'train':
            self.optimizer = self.load_optimizer()
            self.scheduler = ExponentialLR(self.optimizer, gamma=self.args.learning_decay_rate)
            if self.args.resume_training:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(self.checkpoint['scheduler'])
                self.global_step = self.checkpoint['global_step']
                self.start_epoch = self.checkpoint['epoch']
                self.min_error_3d = self.checkpoint['error']
            self.train_writer = SummaryWriter(self.train_log_dir)
            self.test_writer = SummaryWriter(self.test_log_dir)

        self.model_saved_name = os.path.join(self.args.model_saved_path, "model.pth")

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
                                      offset=self.args.offset, random_flip=False, adjust_cube=self.args.adjust_cube)
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
        model = MultiviewA2J(self.dataset_config['camera'], self.num_joints, self.args.n_head, self.args.d_attn,
                             self.args.d_k, self.args.d_v, self.args.d_inner, self.args.dropout_rate,
                             self.args.num_select, use_conf=self.args.use_conf, random_select=self.args.random_select,
                             random_sample=self.args.random_sample)
        self.pre_model_name = self.args.pre_model_name
        if self.pre_model_name is not None:
            if not os.path.exists(self.pre_model_name):
                logger.critical("{} doesn't exist! Please modify your config.".format(self.pre_model_name))
                raise ValueError
            else:
                logger.info("Load model from {}.".format(self.pre_model_name))
                self.checkpoint = torch.load(self.pre_model_name)
                model.load_state_dict(self.checkpoint['model_state_dict'])

        loss_calc = MultiA2JCalculator(self.args.reg_factor, self.args.conf_factor)

        model = nn.DataParallel(model).cuda()
        loss_calc = nn.DataParallel(loss_calc).cuda()
        return model, loss_calc

    def load_optimizer(self):
        optimizer_parameters = []
        for param in self.model.named_parameters():
            optimizer_parameters.append(param[1])
        if self.args.optim == 'Adam':
            optimizer = optim.Adam(
                optimizer_parameters,
                lr=self.args.learning_rate,
                weight_decay=self.args.reg_weight
            )
        elif self.args.optim == 'SGD':
            optimizer = optim.SGD(
                optimizer_parameters,
                lr=self.args.learning_rate,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.args.reg_weight
            )
        else:
            raise NotImplemented
        return optimizer

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch):
        self.model.train()
        logger.info('Training epoch: {}'.format(epoch + 1))
        timer = {'dataloader': 0., 'model': 0., 'statistics': 0.}
        if epoch%(self.args.split-1) == 0:
            np.random.shuffle(self.dataset['train'].index)
        feeder = self.feeder['train'][epoch%(self.args.split-1)]
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        epoch_loss = epoch_anchor_loss = epoch_reg_loss = epoch_conf_loss = epoch_center_error_3d = \
            epoch_min_error_3d = epoch_mean_error_3d = epoch_error_3d_fused = epoch_error_3d_conf = 0.
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
            crop_expand, anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm, joints_3d_pred, \
                view_trans, joint_3d_fused, classification, regression, depthregression, conf, joint_3d_conf = \
                self.model(cropped, crop_trans, com_2d, inter_matrix, cube, self.args.level)
            # print(heatmaps_expand.shape)
            anchor_loss, reg_loss, conf_loss, loss, center_error_3d, min_error_3d, mean_error_3d, _, error_3d_fused, \
            error_3d_conf= self.loss_calc(anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm,
                                          joints_3d_pred, joint_3d_fused, joint_3d_conf, view_trans, crop_trans, com_2d,
                                          cube, self.fx, self.fy, self.u0, self.v0, joint_3d)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            epoch_loss += torch.sum(loss).item()
            epoch_anchor_loss += torch.sum(anchor_loss).item()
            epoch_reg_loss += torch.sum(reg_loss).item()
            epoch_conf_loss += torch.sum(conf_loss).item()
            epoch_center_error_3d += torch.sum(center_error_3d).item()
            epoch_min_error_3d += torch.sum(min_error_3d).item()
            epoch_mean_error_3d += torch.sum(mean_error_3d).item()
            epoch_error_3d_fused += torch.sum(error_3d_fused).item()
            epoch_error_3d_conf += torch.sum(error_3d_conf).item()

            num_sample += cropped.size(0)
            timer['model'] += self.split_time()

            self.train_writer.add_scalar('loss', loss.mean(), global_step=self.global_step)
            self.train_writer.add_scalar('anchor_loss', anchor_loss.mean(), global_step=self.global_step)
            self.train_writer.add_scalar('reg_loss', reg_loss.mean(), global_step=self.global_step)
            self.train_writer.add_scalar('conf_loss', conf_loss.mean(), global_step=self.global_step)
            self.train_writer.add_scalar('center_error_3d', center_error_3d.mean(), global_step=self.global_step)
            self.train_writer.add_scalar('min_error_3d', min_error_3d.mean(), global_step=self.global_step)
            self.train_writer.add_scalar('mean_error_3d', mean_error_3d.mean(), global_step=self.global_step)
            self.train_writer.add_scalar('error_3d_fused', error_3d_fused.mean(), global_step=self.global_step)
            self.train_writer.add_scalar('error_3d_conf', error_3d_conf.mean(), global_step=self.global_step)
            if self.global_step%100==0:
                self.train_writer.add_images('crop_expand', (crop_expand[0][:] + 1.) / 2.,
                                             global_step=self.global_step)
                conf_show = conf[:4]
                conf_show = conf_show / conf_show.max(dim=1)[0][:, None]
                self.train_writer.add_images('conf', conf_show.reshape((4, 1, *self.shape)),
                                             global_step=self.global_step)
            self.global_step += 1
            timer['statistics'] += self.split_time()

        epoch_loss /= num_sample
        epoch_anchor_loss /= num_sample
        epoch_reg_loss /= num_sample
        epoch_conf_loss /= num_sample
        epoch_center_error_3d /= num_sample
        epoch_min_error_3d /= num_sample
        epoch_mean_error_3d /= num_sample
        epoch_error_3d_fused /= num_sample
        epoch_error_3d_conf /= num_sample

        self.scheduler.step()
        timer['model'] += self.split_time()

        lr = self.optimizer.param_groups[0]['lr']
        self.train_writer.add_scalar('lr', lr, self.global_step)
        timer['statistics'] += self.split_time()

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        logger.info('Mean training loss: {:.6f}'.format(epoch_loss))
        logger.info('Mean training anchor_loss: {:.6f}'.format(epoch_anchor_loss))
        logger.info('Mean training reg_loss: {:.6f}'.format(epoch_reg_loss))
        logger.info('Mean training conf_loss: {:.6f}'.format(epoch_conf_loss))
        logger.info('Mean training center_error_3d: {:.2f}mm'.format(epoch_center_error_3d))
        logger.info('Mean training min_error_3d: {:.2f}mm'.format(epoch_min_error_3d))
        logger.info('Mean training mean_error_3d: {:.2f}mm'.format(epoch_mean_error_3d))
        logger.info('Mean training error_3d_fused: {:.2f}mm'.format(epoch_error_3d_fused))
        logger.info('Mean training error_3d_conf: {:.2f}mm'.format(epoch_error_3d_conf))
        logger.info('Time consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

    def eval(self, epoch):
        self.model.eval()
        logger.info('Eval epoch: {}'.format(epoch + 1))
        timer = {'dataloader': 0., 'model': 0., 'statistics': 0.}
        feeder = self.feeder['test']
        self.record_time()
        epoch_loss = epoch_anchor_loss = epoch_reg_loss = epoch_conf_loss = epoch_center_error_3d = \
            epoch_min_error_3d = epoch_mean_error_3d = epoch_error_3d_fused = epoch_error_3d_conf = 0.
        if self.args.level==4:
            num_views = 25
        elif self.args.level==3:
            num_views = 15
        elif self.args.level==2:
            num_views = 9
        elif self.args.level==1:
            num_views = 3
        # epoch_sort_error_3d = np.zeros(25)
        # epoch_error_3d = np.zeros(25)
        # best_num = np.zeros(25, dtype=np.int)
        num_sample = 0
        if self.args.save_result:
            joint_3d_list = []
            joint_2d_list = []
            conf_list = []
            item_list = []
        for batch_idx, data in enumerate(tqdm(feeder, ncols=80)):
            with torch.no_grad():
                item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = data
                timer['dataloader'] += self.split_time()
                cropped = cropped.cuda()
                crop_trans = crop_trans.cuda()
                com_2d = com_2d.cuda()
                inter_matrix = inter_matrix.cuda()
                cube = cube.cuda()
                crop_expand, anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm, joints_3d_pred, \
                view_trans, joint_3d_fused, classification, regression, depthregression, conf, joint_3d_conf = \
                    self.model(cropped, crop_trans, com_2d, inter_matrix, cube, self.args.level)
                anchor_loss, reg_loss, conf_loss, loss, center_error_3d, min_error_3d, mean_error_3d, error_3d, error_3d_fused, \
                error_3d_conf = self.loss_calc(anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm,
                                               joints_3d_pred, joint_3d_fused, joint_3d_conf, view_trans, crop_trans,
                                               com_2d,
                                               cube, self.fx, self.fy, self.u0, self.v0, joint_3d)
                epoch_loss += torch.sum(loss).item()
                epoch_anchor_loss += torch.sum(anchor_loss).item()
                epoch_reg_loss += torch.sum(reg_loss).item()
                epoch_conf_loss += torch.sum(conf_loss).item()
                epoch_center_error_3d += torch.sum(center_error_3d).item()
                epoch_min_error_3d += torch.sum(min_error_3d).item()
                epoch_mean_error_3d += torch.sum(mean_error_3d).item()
                epoch_error_3d_fused += torch.sum(error_3d_fused).item()
                epoch_error_3d_conf += torch.sum(error_3d_conf).item()
                error_3d = error_3d.cpu().numpy()
                # for error in error_3d:
                #     best_id = np.argmin(error)
                #     best_num[best_id] += 1
                #     epoch_error_3d += error
                #     sort_error = np.sort(error)
                #     epoch_sort_error_3d += sort_error
                num_sample += cropped.size(0)
                if self.args.save_result:
                    joint_3d_list.append(joint_3d_conf.cpu().numpy())
                    joint_2d_pred = transform_3D_to_2D(joint_3d_conf, self.fx, self.fy, self.u0, self.v0)
                    joint_2d_list.append(joint_2d_pred.cpu().numpy())
                    conf_list.append(conf.cpu().numpy())
                    item_list.append(item.cpu().numpy())
                timer['model'] += self.split_time()
            if self.args.phase == 'train' and batch_idx % 100 == 0:
                self.test_writer.add_images('crop_expand', (crop_expand[0][:] + 1.) / 2.,
                                            global_step=self.global_step + batch_idx)
                conf_show = conf[:4]
                conf_show = conf_show / conf_show.max(dim=1)[0][:, None]
                self.test_writer.add_images('conf', conf_show.reshape((4, 1, *self.shape)),
                                            global_step=self.global_step + batch_idx)
                timer['statistics'] += self.split_time()

        epoch_loss /= num_sample
        epoch_anchor_loss /= num_sample
        epoch_reg_loss /= num_sample
        epoch_conf_loss /= num_sample
        epoch_center_error_3d /= num_sample
        epoch_min_error_3d /= num_sample
        epoch_mean_error_3d /= num_sample
        # epoch_sort_error_3d /= num_sample
        # epoch_error_3d /= num_sample
        epoch_error_3d_fused /= num_sample
        epoch_error_3d_conf /= num_sample

        timer['model'] += self.split_time()

        if self.args.phase == 'train':
            self.test_writer.add_scalar('loss', epoch_loss, global_step=self.global_step)
            self.test_writer.add_scalar('anchor_loss', epoch_anchor_loss, global_step=self.global_step)
            self.test_writer.add_scalar('reg_loss', epoch_reg_loss, global_step=self.global_step)
            self.test_writer.add_scalar('conf_loss', epoch_conf_loss, global_step=self.global_step)
            self.test_writer.add_scalar('center_error_3d', epoch_center_error_3d, global_step=self.global_step)
            self.test_writer.add_scalar('min_error_3d', epoch_min_error_3d, global_step=self.global_step)
            self.test_writer.add_scalar('mean_error_3d', epoch_mean_error_3d, global_step=self.global_step)
            self.test_writer.add_scalar('error_3d_fused', epoch_error_3d_fused, global_step=self.global_step)
            self.test_writer.add_scalar('error_3d_conf', epoch_error_3d_conf, global_step=self.global_step)
            timer['statistics'] += self.split_time()

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        total_time = timer['dataloader'] + timer['model'] + timer['statistics']

        logger.info('Mean test loss: {:.6f}'.format(epoch_loss))
        logger.info('Mean test anchor_loss: {:.6f}'.format(epoch_anchor_loss))
        logger.info('Mean test reg_loss: {:.6f}'.format(epoch_reg_loss))
        logger.info('Mean test conf_loss: {:.6f}'.format(epoch_conf_loss))
        logger.info('Mean test center_error_3d: {:.2f}mm'.format(epoch_center_error_3d))
        logger.info('Mean test min_error_3d: {:.2f}mm'.format(epoch_min_error_3d))
        logger.info('Mean test mean_error_3d: {:.2f}mm'.format(epoch_mean_error_3d))
        logger.info('Mean test error_3d_fused: {:.2f}mm'.format(epoch_error_3d_fused))
        logger.info('Mean test error_3d_conf: {:.2f}mm'.format(epoch_error_3d_conf))
        # for i in range(len(epoch_sort_error_3d)):
        #     error_3d = epoch_sort_error_3d[i]
        #     logger.info('{}st error_3d: {:.2f}mm'.format(i, error_3d))
        # for i in range(len(epoch_error_3d)):
        #     error_3d = epoch_error_3d[i]
        #     logger.info('{} view error_3d: {:.2f}mm'.format(i, error_3d))
        # for i in range(len(best_num)):
        #     num = best_num[i]
        #     logger.info('{} view best_num: {}'.format(i, num))
        logger.info('Time consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        logger.info('FPS: {:.2f}'.format(num_sample / total_time))

        if self.args.save_result:
            joint_3d_pred = np.concatenate(joint_3d_list, axis=0)
            if self.args.dataset == 'nyu':
                joint_3d_pred[:, :, 1] = -joint_3d_pred[:, :, 1]
            joint_3d_pred = joint_3d_pred.reshape(joint_3d_pred.shape[0], -1)
            joint_2d_pred = np.concatenate(joint_2d_list, axis=0)
            joint_2d_pred = joint_2d_pred.reshape(joint_2d_pred.shape[0], -1)
            conf = np.concatenate(conf_list, axis=0)
            item = np.concatenate(item_list, axis=0)

        if self.args.phase == 'train' and epoch_error_3d_conf < self.min_error_3d:
            self.min_error_3d = epoch_error_3d_conf
            state = {
                'epoch': epoch + 1,
                'global_step': self.global_step,
                'scheduler': self.scheduler.state_dict(),
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'error': epoch_error_3d_conf
            }
            torch.save(state, self.model_saved_name)
            if self.args.save_result:
                np.savetxt(os.path.join(self.args.log_dir, 'joint_3d.txt'), joint_3d_pred, fmt='%.3f')
                np.savetxt(os.path.join(self.args.log_dir, 'joint_2d.txt'), joint_2d_pred, fmt='%.3f')
                np.savetxt(os.path.join(self.args.log_dir, 'conf.txt'), conf, fmt='%.6f')
                np.savetxt(os.path.join(self.args.log_dir, 'item.txt'), item, fmt='%d')

        if self.args.phase == 'eval' and self.args.save_result:
            np.savetxt(os.path.join(self.args.log_dir, 'joint_3d.txt'), joint_3d_pred, fmt='%.3f')
            np.savetxt(os.path.join(self.args.log_dir, 'joint_2d.txt'), joint_2d_pred, fmt='%.3f')
            np.savetxt(os.path.join(self.args.log_dir, 'conf.txt'), conf, fmt='%.6f')
            np.savetxt(os.path.join(self.args.log_dir, 'item.txt'), item, fmt='%d')

    def start(self):
        if self.args.phase == 'train':
            lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', lr, self.global_step)
            for epoch in range(self.start_epoch, self.args.num_epoch):
                self.train(epoch)
                self.eval(epoch)
            logger.info("Min error: {:.2f}mm, model name: {}".format(self.min_error_3d, self.model_saved_name))
        elif self.args.phase == 'eval':
            self.eval(0)


if __name__ == '__main__':
    parser = get_a2j_parser()
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
    if args.phase == 'train' and args.pre_model_name is None and args.resume_training:
        logger.critical('When parameter "pre_model_path" is None, parameter "resume_training" can not be true.')
        raise ValueError('When parameter "pre_model_path" is None, parameter "resume_training" can not be true.')
    if args.phase == 'train':
        init_seed(args.seed)
    else:
        torch.backends.cudnn.benchmark = True
    processor = Processor(args)
    processor.start()