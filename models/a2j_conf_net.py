import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(dir)
sys.path.append(root)
from models.attention import MultiHeadAttention, PositionwiseFeedForward

class A2JConfNet(nn.Module):
    def __init__(self, n_head, d_attn, d_k, d_v, d_inner, dropout_rate, num_select, random=False):
        super(A2JConfNet, self).__init__()
        self.n_head = n_head
        self.d_attn = d_attn
        self.d_k = d_k
        self.d_v = d_v
        self.d_inner = d_inner
        self.dropout_rate = dropout_rate
        self.num_select = num_select
        self.random = random
        self.num_anchors = 11*11*16
        self.encode = nn.Sequential(
            # (B*N*J, 64, 11, 11)
            nn.Conv2d(64, d_attn//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_attn//4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # (B*N*J, d_attn//4, 5, 5)

            nn.Conv2d(d_attn//4, d_attn//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_attn//2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # (B*N*J, d_attn//2, 2, 2)

            nn.Conv2d(d_attn//2, d_attn, kernel_size=2) # (B*N*J, d_attn, 1, 1)
        )
        self.attention = MultiHeadAttention(n_head, d_attn, d_k, d_v, dropout_rate)
        self.pos_ffn = PositionwiseFeedForward(d_attn, d_inner, dropout_rate)
        self.confidence_net = nn.Linear(d_attn, 1)

    def select(self, joint_3d, conf, k, random):
        """

        :param joint_3d: Tensor(B, N, J, 3)
        :param conf: Tensor(B, N)
        :param k: int
        :return:
            conf_select: Tensor(B, k)
            id_select: Tensor(B, k)
        """
        B, N, J, _ = joint_3d.shape
        if random:
            conf_select_list = []
            id_select_list = []
            for i in range(B):
                id = torch.arange(0, N, device=conf.device)
                id = id[torch.randperm(N)]
                id_select = id[:k]

                conf_select = conf[i, id_select]
                conf_select_list.append(conf_select)
                id_select_list.append(id_select)
            conf_select = torch.stack(conf_select_list, dim=0)
            id_select = torch.stack(id_select_list, dim=0)
        conf_select, id_select = torch.topk(conf, k, dim=-1) # (B, k)

        id_select_expand = id_select[:, :, None, None].repeat((1, 1, J, 3))
        joint_3d_select = torch.gather(joint_3d, 1, id_select_expand)  # (B, k, J, 3)

        return joint_3d_select, conf_select, id_select

    def forward(self, classification, regression, depthregression, joint_3d):
        """

        :param classification: Tensor(B, num_views, num_anchors, num_joints)
        :param regression: Tensor(B, num_views, num_anchors, num_joints, 2)
        :param depthregression: Tensor(B, num_views, num_anchors, num_joints)
        :param joint_3d: Tensor(B, num_views, num_joints, 3)
        :return:
        """
        B, N, J, _ = joint_3d.shape
        # (B, N, num_anchors, num_joints, 4)
        input = torch.cat([classification[..., None], regression, depthregression[..., None]], dim=-1)
        input = torch.transpose(input, 2, 3) # # (B, N, J, num_anchors, 4)
        input = torch.reshape(input, (B*N*J, 11, 11, 16*4))
        input = input.transpose(1, 3).transpose(2, 3) # (B*N*J, 64, 11, 11)
        feature = self.encode(input).reshape([B, N, J, -1]) # (B, N, J, d_attn)
        v = feature.mean(dim=-2)  # (B, N,d_attn)

        v = self.attention(v, v, v)
        v = self.pos_ffn(v)

        conf = self.confidence_net(v).reshape([B, N])
        joint_3d_select, conf_select, id_select = self.select(joint_3d, conf, self.num_select, self.random)

        conf_select = torch.softmax(conf_select, dim=-1)  # (B, k)

        joint_3d_conf = joint_3d_select * conf_select[:, :, None, None]
        joint_3d_conf = torch.sum(joint_3d_conf, 1)

        return conf, joint_3d_conf
