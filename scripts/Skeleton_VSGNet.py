from __future__ import print_function, division
import torch
import torch.nn as nn

import os
import numpy as np
import pool_pairing  as ROI

import torchvision.models as models

lin_size = 1024
ids = 80
context_size = 1024
sp_size = 1024
mul = 3
deep = 512
pool_size = (10, 10)
pool_size_pose = (18, 5, 5)
num_node = 26


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)

class embed(nn.Module):
    def __init__(self, dim=3, dim1=128, bias=False):
        super(embed, self).__init__()

        self.cnn = nn.Sequential(
            nn.BatchNorm1d(dim),
            cnn1x1(dim, 128, bias=bias),
            nn.ReLU(),
            cnn1x1(128, dim1, bias=bias),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


class compute_g_spa(nn.Module):
    def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        g1 = self.g1(x1).permute(0, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g


class cnn1x1(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv1d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x


class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm1d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):
        x = x1.permute(0, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x

class VSGNet(nn.Module):
    def __init__(self):
        super(VSGNet, self).__init__()

        def one_hot(spa):
            y = torch.arange(spa).unsqueeze(-1)
            y_onehot = torch.FloatTensor(spa, spa)

            y_onehot.zero_()
            y_onehot.scatter_(1, y, 1)

            y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
            y_onehot = torch.reshape(y_onehot, (1, spa, spa))
            # y_onehot = y_onehot.repeat(bs, 1, 1)

            return y_onehot

        model = models.resnet152(pretrained=True)
        self.flat = Flatten()

        ###################################graph##############################
        self.keypoint_embed = embed(2, 64)
        self.conv = cnn1x1(64, 512, bias=False)
        self.compute_g1 = compute_g_spa(64, 256)
        self.gcn1 = gcn_spa(64, 256)
        self.gcn2 = gcn_spa(256, 256)
        self.gcn3 = gcn_spa(256, 512)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.skeleton_tail = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 29),
            nn.ReLU()
        )

        self.lin_graph_head = nn.Sequential(
            # nn.Linear(2048, 29),
            # nn.Dropout2d(p=0.5),
            nn.Linear(lin_size * 2, 1024),
            nn.Linear(1024, 512),

            nn.ReLU(),

        )
        self.lin_graph_tail = nn.Sequential(
            nn.Linear(512, 29),

        )

        # self.spa_onehot = one_hot(num_node)
        # self.onehot_embed = embed(num_node, 64)



        self.Conv_pretrain = nn.Sequential(*list(model.children())[0:7])  ## Resnets,resnext

        ######### Convolutional Blocks for human,objects and the context##############################
        self.Conv_people = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.Conv_objects = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.Conv_context = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )

        ###############################################################################################

        ##### Attention Feature Model######
        self.conv_sp_map = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(5, 5)),
            # nn.Conv2d(3, 64, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 32, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.AvgPool2d((13, 13), padding=0, stride=(1, 1)),
            # nn.Linear(32,1024),
            # nn.ReLU()

        )
        self.spmap_up = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(),
        )

        #######################################

        ### Prediction Model for attention features#######
        self.lin_spmap_tail = nn.Sequential(
            nn.Linear(512, 29),

        )


        #################################################################

        # Interaction prediction model for visual features######################
        self.lin_single_head = nn.Sequential(
            # nn.Linear(2048,1),
            # nn.Dropout2d(p=0.5),
            nn.Linear(lin_size * 3 + 4, 1024),
            # nn.Linear(lin_size*3, 1024),
            nn.Linear(1024, 512),

            nn.ReLU(),

        )
        self.lin_single_tail = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(512, 1),
            # nn.Linear(10,1),

        )

        #################################################################

        ########## Prediction model for visual features#################

        self.lin_visual_head = nn.Sequential(
            # nn.Linear(2048, 29),
            # nn.Dropout2d(p=0.5),
            nn.Linear(lin_size * 3 + 4, 1024),
            # nn.Linear(lin_size*3, 1024),
            # nn.Linear(lin_size*3+4+sp_size, 1024),
            nn.Linear(1024, 512),

            nn.ReLU(),
            #  nn.ReLU(),
        )
        self.lin_visual_tail = nn.Sequential(
            nn.Linear(512, 29),

        )

        ################################################

        ####### Prediction model for graph features##################
        self.lin_graph_head = nn.Sequential(
            # nn.Linear(2048, 29),
            # nn.Dropout2d(p=0.5),
            nn.Linear(512, 256),
            nn.Linear(256, 128),

            nn.ReLU(),

        )
        self.lin_graph_tail = nn.Sequential(
            nn.Linear(128, 29),

        )

        self.fc = nn.Sequential(
            nn.Linear(13312, 512)
        )

        ########################################

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pairs_info, pairs_info_augmented, image_id, flag_, phase):
        out1 = self.Conv_pretrain(x)  ###

        rois_people, rois_objects, spatial_locs, union_box, keypoints = ROI.get_pool_loc(out1, image_id, flag_, size=pool_size,
                                                                              spatial_scale=25,
                                                                              batch_size=len(pairs_info))

        ################keypoints####################
        keypoints = keypoints.permute(0, 2, 1)
        # bs = keypoints.size(0)

        # onehot_emd = self.spa_onehot.repeat(bs, 1, 1)
        # onehot_output = self.onehot_embed(onehot_emd.cuda())
        keypoints_embed = self.keypoint_embed(keypoints)

        # keypoints_embed = torch.cat([keypoints_embed, onehot_output], 1)
        g = self.compute_g1(keypoints_embed)
        a1 = nn.Parameter(torch.ones(g.size())).cuda()
        a2 = nn.Parameter(torch.ones(g.size())).cuda()
        a3 = nn.Parameter(torch.ones(g.size())).cuda()
        keypoints_out = self.gcn1(keypoints_embed, g + a1)
        keypoints_out = self.gcn2(keypoints_out, g + a2)
        keypoints_out = self.gcn3(keypoints_out, g + a3)
        # keypoints_out = self.maxpool(keypoints_out)
        res_keypoints = self.conv(keypoints_embed)
        # keypoints_out = torch.cat([keypoints_out, res_keypoints], 1)
        keypoints_out = keypoints_out + res_keypoints
        keypoints_out = torch.flatten(keypoints_out, 1)
        # keypoints_out = self.fc(keypoints_out)


        ### Defining The Pooling Operations #######
        x, y = out1.size()[2], out1.size()[3]
        hum_pool = nn.AvgPool2d(pool_size, padding=0, stride=(1, 1))
        obj_pool = nn.AvgPool2d(pool_size, padding=0, stride=(1, 1))
        context_pool = nn.AvgPool2d((x, y), padding=0, stride=(1, 1))
        #################################################

        ### Human###
        residual_people = rois_people
        res_people = self.Conv_people(rois_people) + residual_people
        res_av_people = hum_pool(res_people)
        out2_people = self.flat(res_av_people)
        ###########

        ##Objects##
        residual_objects = rois_objects
        res_objects = self.Conv_objects(rois_objects) + residual_objects
        res_av_objects = obj_pool(res_objects)
        out2_objects = self.flat(res_av_objects)
        #############

        #### Context ######
        residual_context = out1
        res_context = self.Conv_context(out1) + residual_context
        res_av_context = context_pool(res_context)
        out2_context = self.flat(res_av_context)
        #################

        ##Attention Features##
        out2_union = self.spmap_up(self.flat(self.conv_sp_map(union_box)))
        ############################

        #### Making Essential Pairing##########
        pairs, people, objects_only = ROI.pairing(out2_people, out2_objects, out2_context, spatial_locs, pairs_info)
        ####################################

        ###### Interaction Probability##########
        lin_single_h = self.lin_single_head(pairs)
        lin_single_t = lin_single_h * out2_union
        lin_single = self.lin_single_tail(lin_single_t)
        interaction_prob = self.sigmoid(lin_single)
        ####################################################


        ######################################################################################################################################

        #### Prediction from visual features####
        lin_h = self.lin_visual_head(pairs)
        lin_t = lin_h * out2_union
        lin_visual = self.lin_visual_tail(lin_t)
        ##############################



        ####################################

        ##### Prediction from attention features #######
        lin_att = self.lin_spmap_tail(out2_union)
        #############################

        ########Prediction from graph features#########
        # lin_graph = self.skeleton_tail(keypoints_out)
        # graph = torch.cat((self.fc(keypoints_out), lin_h), 1)
        graph = self.fc(keypoints_out) * lin_h
        lin_graph_h = self.lin_graph_head(graph)
        # lin_graph_t = lin_graph_h * out2_union
        lin_graph = self.lin_graph_tail(lin_graph_h)

        return [lin_visual, lin_single, lin_graph, lin_att]  # ,lin_obj_ids]
