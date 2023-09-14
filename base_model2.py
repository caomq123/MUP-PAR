import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from backbone import *
from utils import *
from roi_align.roi_align import RoIAlign  # RoIAlign module
from roi_align.roi_align import CropAndResize  # crop_and_resize module
from self_attention_cv import AxialAttentionBlock


class Basenet_collective2(nn.Module):
    """
    main module of base model for collective dataset
    """

    def __init__(self, cfg):
        super(Basenet_collective2, self).__init__()
        self.cfg = cfg

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        self.backbone = MyResnet_18()
        #         self.backbone=MyVGG16(pretrained=True)

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)

        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.dropout_emb_1 = nn.Dropout(p=self.cfg.train_dropout_prob)
        #         self.nl_emb_1=nn.LayerNorm([NFB])

        self.fc_actions = nn.Linear(NFB, self.cfg.num_actions)
        self.fc_activities = nn.Linear(NFB, self.cfg.num_activities)
        self.attention = AxialAttentionBlock(in_channels=D, dim=K, heads=8)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def savemodel(self, filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'fc_emb_state_dict': self.fc_emb_1.state_dict(),
            'fc_actions_state_dict': self.fc_actions.state_dict(),
            'fc_activities_state_dict': self.fc_activities.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in = batch_data
        # images_in [4, 1, 3, 480, 720]  [B, T, C, H, W]
        # boxes_in [4, 1, 60, 4]
        # bboxes_num_in [[31],[36],[14],[15]]       因为batchsize为4-有4个image-每个image有[31/36/14/15]个person框

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size     # [57, 87]
        MAX_N = self.cfg.num_boxes     # 60
        NFB = self.cfg.num_features_boxes  # 1024
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn  # 256, 1024
        EPS = 1e-5

        D = self.cfg.emb_features    # 1056
        K = self.cfg.crop_size[0]    # (5, 5)

        # Reshape the input data--这个作者没有考虑时间维度T 所以T就是1
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W---[4, 3, 480, 720]
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)   # [4, 60, 4]

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)   # [4, 3, 480, 720]
        outputs = self.backbone(images_in_flat)        # cnn抽取输入图片的特征（一整张图片吗？对）
        # print(outputs.shape)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:  # [4, 288, 57, 87]  [4, 768, 28, 43]
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)  # [4, 288, 57, 87]  [4, 768, 57, 87]
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW
        # print(features_multiscale.shape)

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4    # 每张图片对应的box框数吗？
        # [240, 4]=[4*1*60, 4]
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]  # torch.ones和torch.zeros一样初始化输出
        # [4, 60](全1,全2,全3,全4)  batch_size是4
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N   # stack属于扩张再连接，扩出一个维度
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,  为什么要初始化这个？

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False  # [240, 1056, 5, 5]
        boxes_features_all = self.roi_align(features_multiscale,   # 输入的每个image
                                            boxes_in_flat,   # 输入的每个image中包含的person box
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K, 输出的是每个image中每个person的D*K*K的特征？
        # self attention [240, 1056, 5, 5]  240 = 4*1*60
        boxes_features_all = boxes_features_all.reshape(B * T * MAX_N, D, K, K)  # B*T*MAX_N, D,K,K
        boxes_features_all = self.attention(boxes_features_all)     # 计算所有person的注意力之后的特征呀 [240, 1056, 5, 5]
        boxes_features_all = boxes_features_all.reshape(B * T, MAX_N, -1)  # B*T,MAX_N, D*K*K  [4, 60, 26400]


        # Embedding  维度不变[4, 60, 1024]
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all = F.relu(boxes_features_all)
        boxes_features_all = self.dropout_emb_1(boxes_features_all)      # embedding有啥意义捏~ 维度映射一下？高维映射到低维？

        actions_scores = []
        activities_scores = []
        bboxes_num_in = bboxes_num_in.reshape(B * T, )  # B*T,        一共多少个image B*T个

        for bt in range(B * T):  # 一张图片一张图片的来作识别
            N = bboxes_num_in[bt]       # 第bt个pearson框box N=31, 36, 14, 15
            boxes_features = boxes_features_all[bt, :N, :].reshape(1, N, NFB)  # 1,N,NFB
            # [1, 31, 1024][1, 36, 1024][1, 14, 1024][1, 15, 1024]
            boxes_states = boxes_features   # 这啥玩意意思嘞

            NFS = NFB  # 1024

            # Predict actions
            boxes_states_flat = boxes_states.reshape(-1, NFS)  # 1*N, NFS--[31, 1024]
            actn_score = self.fc_actions(boxes_states_flat)  # 1*N, actn_num---[31, 27]
            actions_scores.append(actn_score)

            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim=1)  # 1, NFS--[1, 1024]
            boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, NFS)  # 1, NFS--[1, 1024]
            acty_score = self.fc_activities(boxes_states_pooled_flat)  # 1, acty_num--[1, 7]
            activities_scores.append(acty_score)

        actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num  [96, 27] (31+36+14+15) 多少个人就多少个动作
        activities_scores = torch.cat(activities_scores, dim=0)  # B*T,acty_num [4, 7]

        #         print(actions_scores.shape)
        #         print(activities_scores.shape)

        return actions_scores, activities_scores
