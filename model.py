#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import sys
from scipy.spatial import cKDTree
import pickle
from scipy import stats
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

EPS = 1e-6


def knn(x, k):
    # distance between (xi, yi, zi) and (xj, yj, zj) is
    # sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2) =
    # sqrt((xi ** 2 + yi ** 2 + zi ** 2) + (xj ** 2 + yj ** 2 + zj ** 2) -2 (xixj + yiyj + zizj)) =
    # Since we only care about ordering
    # (xi ** 2 + yi ** 2 + zi ** 2) + (xj ** 2 + yj ** 2 + zj ** 2) -2 (xixj + yiyj + zizj)

    # calculate -2 (xixj + yiyj + zizj) term
    inner = -2 * torch.matmul(x.transpose(2, 1), x)

    # calculate (xi ** 2 + yi ** 2 + zi ** 2) terms
    xx = torch.sum(x ** 2, dim=1, keepdim=True)

    # negates distance for topk function
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # gets topk indices
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None, use_x_coord=False):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if not use_x_coord:  # dynamic knn graph
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:  # fixed knn graph with input point coordinates
            x_coord = x_coord.view(batch_size, -1, num_points)
            # if we just do idx = knn(x_coord, k=k), we get nan loss
            idx = knn(x_coord, k=k + 1)
            idx = idx[:, :, 1:]  # find k nearest neighbors for each point (excluding self as negihbor)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature


class VNEmbedding(nn.Module):
    def __init__(self):
        super(VNEmbedding, self).__init__()

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        batch_size = x.size(0)
        num_points = x.size(3)
        x = x.view(batch_size, -1, num_points)
        _, feature_num_dims, _ = x.size()
        feature_num_dims = feature_num_dims // 3

        # calculate distance matrix
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        concat_x = torch.unsqueeze(x, 3)
        concat_x = concat_x.transpose(2, 1).contiguous()

        for k in [8, 16, 32]:
            idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

            device = torch.device('cuda')

            # indices in each batch are separate
            # we will now reindex and combine indices across batches
            idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            idx = idx + idx_base
            idx = idx.view(-1)

            # get points corresponding to idx
            x = x.transpose(2,
                            1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
            feature = x.view(batch_size * num_points, -1)[idx, :]

            # take average
            feature = feature.view(batch_size, num_points, k, feature_num_dims, 3)
            feature = torch.mean(feature, 2, keepdim=True)

            num_dims = concat_x.size()[3]
            concat_x = concat_x.view(batch_size, num_points, 1, num_dims, 3)
            concat_x = torch.cat((feature, concat_x), dim=3).permute(0, 4, 1, 3, 2).contiguous()

        concat_x = concat_x.permute(0, 3, 1, 2, 4).contiguous()
        return concat_x


def vn_get_keypoints(a, f):
    '''
    a: point features of shape [B, K, P]
    f: point features of shape [B, C, P]
    '''

    af = torch.unsqueeze(a, 2) * torch.unsqueeze(f, 1)  # B x K x C x P
    num = af.permute(3, 2, 0, 1)  # P x C x B x K

    sum_a = torch.sum(a, 2)  # B x K
    denom = sum_a + EPS

    div = num / denom  # P x C x B x K
    mu = torch.sum(div, 0).permute(1, 2, 0)  # B x K x C

    return mu


class VNRegression(nn.Module):
    def __init__(self, in_channels):
        super(VNRegression, self).__init__()
        self.w1 = nn.Linear(in_channels * in_channels, 1, bias=False)
        self.w2 = nn.Linear(in_channels * in_channels, 1, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, K, C]
        '''
        B, K, C = x.size()
        x = torch.unsqueeze(x, 2)  # B x K x 1 x C
        pairwise_difference = x - torch.transpose(x, 1, 2)  # B x K x K x C
        pairwise_difference = torch.reshape(pairwise_difference, (B, K * K, C))  # B x K * K x C
        v1 = torch.squeeze(self.w1(pairwise_difference.transpose(1, -1)), 2)  # B x C
        norm_v1 = torch.sqrt(torch.sum(v1 * v1, 1) + EPS)  # B
        v1 = torch.transpose(torch.transpose(v1, 0, 1) / norm_v1, 0, 1)  # B x C
        v2 = torch.squeeze(self.w2(pairwise_difference.transpose(1, -1)), 2)  # B x C
        norm_v2 = torch.sqrt(torch.sum(v2 * v2, 1) + EPS)  # B
        v2 = torch.transpose(torch.transpose(v2, 0, 1) / norm_v2, 0, 1)  # B x C
        v3 = torch.cross(v1, v2, dim=1)
        v1 = torch.unsqueeze(v1, 1)
        v2 = torch.unsqueeze(v2, 1)
        v3 = torch.unsqueeze(v3, 1)
        r = torch.cat((v1, v2, v3), 1)
        return r


def dot_prod_attention_weights(Q, K):
    '''
    Q: queries of shape [..., P, H, E, 3]
    K: keys of shape [..., P', H, E, 3]
    '''

    # calculate dot product
    reshaped_K = torch.transpose(K, -1, -2)  # B x P' x H x 3 x E
    dot = torch.matmul(torch.unsqueeze(Q, -4), torch.unsqueeze(reshaped_K, -5))  # B x P x P' x H x E x E

    # sum over E dimension
    dim = dot.size()[-1]
    size = list(dot.size())[:-2]
    reshaped_dot = torch.einsum('bii->b', torch.reshape(dot, (-1, dim, dim)))  # B * P * P' * H
    summed_dot = torch.reshape(reshaped_dot, size)  # B x P x P' x H
    summed_dot = torch.transpose(torch.transpose(summed_dot, -1, -2), -2, -3)  # B x H x P x P'

    # calculate softmax
    a = torch.nn.functional.softmax(summed_dot, dim=-1)  # B x H x P x P'

    return a


def cross_prod_attention_weights(Q, K):
    '''
    Q: queries of shape [B, C, H, E, 3]
    K: keys of shape [B, C', H, E, 3]
    '''

    # calculate norm of cross product
    b, c, h, e, _ = Q.size()
    _, c_prime, _, _, _ = K.size()
    reshaped_Q = torch.unsqueeze(Q, 2)  # B x C x 1 x H x E x 3
    reshaped_Q = reshaped_Q.expand(-1, -1, c_prime, -1, -1, -1)  # B x C x C' x H x E x 3
    reshaped_K = torch.unsqueeze(K, 1)  # B x 1 x C' x H x E x 3
    reshaped_K = reshaped_K.expand(-1, c, -1, -1, -1, -1)  # B x C x C' x H x E x 3
    cross = torch.cross(reshaped_Q, reshaped_K, dim=-1)  # B x C x C' x H x E x 3
    norm = torch.linalg.norm(cross, dim=-1)  # B x C x C' x H x E

    # sum over E dimension
    summed_dot = torch.mean(norm, dim=-1)  # B x C x C' x H
    summed_dot = summed_dot.permute(0, 3, 1, 2)  # B x H x C x C'
    summed_dot *= 100

    # calculate softmax
    a = torch.nn.functional.softmax(summed_dot, dim=-1)  # B x H x C x C'

    return a


def orthonormal_value(Q, K):
    '''
    Q: queries of shape [B, C, H, E, 3]
    K: keys of shape [B, C', H, E, 3]
    '''

    b, c, h, e, _ = Q.size()
    _, c_prime, _, _, _ = K.size()
    reshaped_Q = torch.unsqueeze(Q, 2)  # B x C x 1 x H x E x 3
    reshaped_Q = reshaped_Q.expand(-1, -1, c_prime, -1, -1, -1)  # B x C x C' x H x E x 3
    reshaped_K = torch.unsqueeze(K, 1)  # B x 1 x C' x H x E x 3
    reshaped_K = reshaped_K.expand(-1, c, -1, -1, -1, -1)  # B x C x C' x H x E x 3

    dot = torch.matmul(torch.unsqueeze(reshaped_Q, 5), torch.unsqueeze(reshaped_K, 6))  # B x C x C' x H x E x 1 x 1
    dot = torch.squeeze(torch.squeeze(dot, 6), 5)  # B x C x C' x H x E

    q_norm = torch.linalg.norm(reshaped_Q, axis=5)  # B x C x C' x H x E
    k_norm = torch.linalg.norm(reshaped_K, axis=5)  # B x C x C' x H x E

    dot = dot / (q_norm * k_norm + EPS)  # B x C x C' x H x E

    dot[dot > (1 - EPS)] = 1 - EPS  # B x C x C' x H x E
    dot[dot < (-1 + EPS)] = -1 + EPS  # B x C x C' x H x E

    acos = torch.arccos(dot)  # B x C x C' x H x E
    dev = torch.abs((np.pi / 2) - acos)  # B x C x C' x H x E
    dev = torch.mean(dev, 4).permute(0, 3, 1, 2)  # B x H x C x C'

    # calculate softmax
    a = -torch.nn.functional.softmax(dev, dim=-1)  # B x H x C x C'

    return a


def attention_layer(Q, K, V, type):
    '''
    Q: queries of shape [B, P, H, E, 3]
    K: keys of shape [B, P, H, E, 3]
    V: values of shape [B, P, H, E', 3]
    '''

    # get attention weights
    if type == 'dot':
        A = dot_prod_attention_weights(Q, K)  # B x H x P x P'
    elif type == 'cross':
        A = cross_prod_attention_weights(Q, K)  # B x H x P x P'

    # multiply attention weights and values
    reshaped_A = torch.transpose(torch.transpose(A, -2, -3), -1, -2)  # B x P x P' x H
    reshaped_A = torch.unsqueeze(torch.unsqueeze(reshaped_A, -1), -1)  # B x P x P' x H x 1 x 1
    prod = reshaped_A * torch.unsqueeze(V, -5)  # B x P x P' x H x E' x 3
    # sum over P' dimension
    att = torch.sum(prod, -4)  # B x P x H x E' x 3
    return att


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels * 2, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_mean = x.mean(1, keepdim=True).expand(x.size())
        x_cross = torch.cross(x, x_mean, dim=2)
        x_cross = x_cross.permute(0, 1, 3, 4, 2)
        x_cross = x_cross.reshape((-1, 3))
        x_cross = torch.abs(x_cross).sum(0)
        print(x.size(), x_cross.size())
        x = torch.cat([x, x_cross], dim=1)
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNSimpleLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNSimpleLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNReflectionLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNReflectionLinear, self).__init__()
        self.A = nn.Linear(in_channels * 2, out_channels, bias=False)
        self.B = nn.Linear(in_channels * 2, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, in_channels, 3, N_samples, ...]
        '''
        x_mean = x.mean(1, keepdim=True).expand(x.size())
        x_cross = torch.cross(x, x_mean, dim=2)
        x = torch.cat([x, x_cross], dim=1)

        avn = self.A(x.transpose(1, -1)).transpose(1, -1)  # B x out_channels x 3 x N_samples

        un = torch.roll(x, 1, dims=1)
        cross = torch.cross(un, x, dim=2)  # B x in_channels x 3 x N_samples
        norm = torch.sqrt((un * un).sum(2, keepdim=True) + EPS)  # B x in_channels x N_samples
        cross = (cross.transpose(2, 1).transpose(1, 0) / norm).transpose(1, 0).transpose(2,
                                                                                         1)  # B x in_channels x 3 x N_samples
        bcross = self.B(cross.transpose(1, -1)).transpose(1, -1)  # B x out_channels x 3 x N_samples
        output = avn + bcross  # B x out_channels x 3 x N_samples
        return output


def get_on_vector(normalized_J):
    '''
    normalized_J: normalized direction vector [batch size, num points, embedding dimension, 3]
    '''

    # calculate normalized U vector that is orthogonal to J
    # Let J = [x, y, z]
    # Then U = [x, y, -(x^2 + y^2) / (z + eps)]

    # get [x, y]
    sub_vec_J = normalized_J[:, :, :, :2]  # b x c x e x 2

    edge_fix = torch.zeros(sub_vec_J.size())
    edge_fix[:,:,:,0] = 1
    mask = sub_vec_J.sum(dim=-1) == 0
    opp_mask = sub_vec_J.sum(dim=-1) != 0
    sub_vec_J = sub_vec_J + edge_fix * mask

    # calculate (x^2 + y^2)
    U_z = torch.squeeze(torch.squeeze(torch.matmul(sub_vec_J.unsqueeze(3), sub_vec_J.unsqueeze(4)), 4), 3)  # b x c x e

    # calculate -(x^2 + y^2) / (z + eps)
    U_z = -U_z / (normalized_J[:, :, :, 2] + EPS)  # b x c x e
    U_z = U_z * opp_mask

    # form [x, y, -(x^2 + y^2) / (z + eps)]
    U = torch.cat((sub_vec_J, U_z.unsqueeze(3)), dim=3)  # b x c x e x 3

    # normalize
    normalized_U = (U.permute(3, 0, 1, 2) / (torch.linalg.norm(U, dim=3) + EPS)).permute(1, 2, 3, 0)  # b x c x e x 3

    return normalized_U


def get_basis(J):
    '''
    J: direction vector [batch size (B), num points (C), embedding dimension (E), 3]
    '''

    # normalize J vectors
    normalized_J = (J.permute(3, 0, 1, 2) / (torch.linalg.norm(J, dim=3) + EPS)).permute(1, 2, 3, 0)  # b x c x e x 3

    normalized_U = get_on_vector(normalized_J)  # b x c x e x 3

    # calculate V vector that is orthogonal to J and U
    normalized_V = torch.cross(normalized_U, normalized_J, dim=3)  # b x c x e x 3

    # R = (U, V, J)
    R = torch.cat((normalized_U, normalized_V, normalized_J), dim=-1)  # b x c x e x 9
    B, C, E, _ = R.size()
    R = torch.reshape(R, (B, C, E, 3, 3))  # b x c x e x 3 x 3

    return R


def get_rtx(index, RT, X):
    '''
    R: rotation basis [b, c, e, 3, 3]
    X: point features of shape [B, C, E, 3]
    '''

    indexed_RT = RT[:, :, :, index, :]  # b x c x e x 3
    rtx = torch.matmul(torch.unsqueeze(indexed_RT, 3), torch.unsqueeze(X, 4))  # b x c x e x 1 x 1
    rtx = torch.squeeze(torch.squeeze(rtx, 4), 3)  # b x c x e

    return rtx


def get_rrtx(index, R, RTX):
    '''
    R: rotation basis [b, c, e, 3, 3]
    RTX: [B, C, E]
    '''

    indexed_R = R[:, :, :, :, index].permute(3, 0, 1, 2)  # 3 x b x c x e
    rrtx = (indexed_R * RTX).permute(1, 2, 3, 0)  # b x c x e x 3

    return rrtx


class ComplexLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexLinear, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J, device):
        '''
        X: point features of shape [..., C, E, 3]
        J: directions of shape [..., C, E, 3] (same shape as X)
        '''

        R = get_basis(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3

        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3

        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
                    mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # Conv
        self.map_to_feat = nn.Linear(2 * in_channels, out_channels, bias=False)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

        # LeakyReLU
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(2 * in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(2 * in_channels, out_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # x_mean = x.mean(1, keepdim=True).expand(x.size())

        # x_cross = torch.cross(x, x_mean, dim=2)
        # x = torch.cat([x, x_cross], dim=1)
        # Conv
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
                    mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNReflectionLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(VNReflectionLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # Conv
        self.A = nn.Linear(2 * in_channels, out_channels, bias=False)
        self.B = nn.Linear(2 * in_channels, out_channels, bias=False)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

        # LeakyReLU
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(2 * in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(2 * in_channels, out_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, in_channels, 3, N_samples, ...]
        '''
        x_mean = x.mean(1, keepdim=True).expand(x.size())
        x_cross = torch.cross(x, x_mean, dim=2)
        x = torch.cat([x, x_cross], dim=1)

        # Conv
        avn = self.A(x.transpose(1, -1)).transpose(1, -1)  # B x out_channels x 3 x N_samples

        un = torch.roll(x, 1, dims=1)
        cross = torch.cross(un, x, dim=2)  # B x in_channels x 3 x N_samples
        norm = torch.sqrt((un * un).sum(dim=2) + EPS)  # B x in_channels x N_samples
        cross = (cross.transpose(2, 1).transpose(1, 0) / norm).transpose(1, 0).transpose(2,
                                                                                         1)  # B x in_channels x 3 x N_samples
        bcross = self.B(cross.transpose(1, -1)).transpose(1, -1)  # B x out_channels x 3 x N_samples
        p = avn + bcross  # B x out_channels x 3 x N_samples

        # InstanceNorm
        if self.use_batchnorm != 'none':
            p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
                    mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNReflectionLinearLeakyReLUBOnly(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(VNReflectionLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # Conv
        self.B = nn.Linear(2 * in_channels, out_channels, bias=False)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

        # LeakyReLU
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(2 * in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(2 * in_channels, out_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, in_channels, 3, N_samples, ...]
        '''
        x_mean = x.mean(1, keepdim=True).expand(x.size())
        x_cross = torch.cross(x, x_mean, dim=2)
        x = torch.cat([x, x_cross], dim=1)

        # Conv
        un = torch.roll(x, 1, dims=1)
        cross = torch.cross(un, x, dim=2)  # B x in_channels x 3 x N_samples
        norm = torch.sqrt((un * un).sum(dim=2) + EPS)  # B x in_channels x N_samples
        cross = (cross.transpose(2, 1).transpose(1, 0) / norm).transpose(1, 0).transpose(2,
                                                                                         1)  # B x in_channels x 3 x N_samples
        bcross = self.B(cross.transpose(1, -1)).transpose(1, -1)  # B x out_channels x 3 x N_samples
        p = bcross  # B x out_channels x 3 x N_samples

        # InstanceNorm
        if self.use_batchnorm != 'none':
            p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
                    mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(VNLinearAndLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x_lin = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x_bn = self.batchnorm(x_lin)
        # LeakyReLU
        x_out = self.leaky_relu(x_bn)
        return x_out


class VNSimpleLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(VNSimpleLinearAndLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = VNSimpleLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class VNReflectionLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(VNReflectionLinearAndLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = VNReflectionLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class ComplexLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(ComplexLinearAndLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = ComplexLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [..., C, E, 3]
        J: directions of shape [..., C, E, 3] (same shape as X)
        '''
        x = self.linear(X, J, device).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim, mode='norm'):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        self.mode = mode
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        norm = torch.sqrt((x * x).sum(2) + EPS)
        if self.mode == 'norm':
            norm_bn = self.bn(norm)
        elif self.mode == 'norm_log':
            norm_log = torch.log(norm)
            norm_log_bn = self.bn(norm_log)
            norm_bn = torch.exp(norm_log_bn)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / (norm + EPS) * norm_bn

        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, n_layers=3, normalize_frame=False, share_nonlinearity=False,
                 use_batchnorm=True, combine_lin_nonlin=True, reflection_variant=True, b_only=False):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.n_layers = n_layers

        if n_layers == 3:
            if combine_lin_nonlin:
                if reflection_variant:
                    if b_only:
                        self.vn1 = VNReflectionLinearLeakyReLUBOnly(in_channels, in_channels // 2, dim=dim,
                                                                    share_nonlinearity=share_nonlinearity,
                                                                    use_batchnorm=use_batchnorm)
                        self.vn2 = VNReflectionLinearLeakyReLUBOnly(in_channels // 2, in_channels // 4, dim=dim,
                                                                    share_nonlinearity=share_nonlinearity,
                                                                    use_batchnorm=use_batchnorm)
                    else:
                        self.vn1 = VNReflectionLinearLeakyReLU(in_channels, in_channels // 2, dim=dim,
                                                               share_nonlinearity=share_nonlinearity,
                                                               use_batchnorm=use_batchnorm)
                        self.vn2 = VNReflectionLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim,
                                                               share_nonlinearity=share_nonlinearity,
                                                               use_batchnorm=use_batchnorm)
                else:
                    self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim,
                                                 share_nonlinearity=share_nonlinearity, use_batchnorm=use_batchnorm)
                    self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim,
                                                 share_nonlinearity=share_nonlinearity, use_batchnorm=use_batchnorm)
            else:
                if reflection_variant:
                    self.vn1 = VNReflectionLinearAndLeakyReLU(in_channels, in_channels // 2, dim=dim,
                                                              share_nonlinearity=share_nonlinearity,
                                                              use_batchnorm=use_batchnorm)
                    self.vn2 = VNReflectionLinearAndLeakyReLU(in_channels // 2, in_channels // 4, dim=dim,
                                                              share_nonlinearity=share_nonlinearity,
                                                              use_batchnorm=use_batchnorm)
                else:
                    self.vn1 = VNLinearAndLeakyReLU(in_channels, in_channels // 2, dim=dim,
                                                    share_nonlinearity=share_nonlinearity, use_batchnorm=use_batchnorm)
                    self.vn2 = VNLinearAndLeakyReLU(in_channels // 2, in_channels // 4, dim=dim,
                                                    share_nonlinearity=share_nonlinearity, use_batchnorm=use_batchnorm)
            if normalize_frame:
                self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
            else:
                self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)
        elif n_layers == 1:
            if normalize_frame:
                self.vn_lin = nn.Linear(in_channels, 2, bias=False)
            else:
                self.vn_lin = nn.Linear(in_channels, 3, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        if self.n_layers == 3:
            z0 = self.vn1(z0)
            z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:, 0, :]
            # u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True) + EPS)
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            # u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True) + EPS)
            u2 = v2 / (v2_norm + EPS)

            # compute the cross product of the two output vectors
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)

        return x_std, z0


class VNSimpleStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, share_nonlinearity=False, use_batchnorm=True, combine_lin_nonlin=True,
                 reflection_variant=True, b_only=False):
        super(VNSimpleStdFeature, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm

        self.vn1 = VNSimpleLinearAndLeakyReLU(in_channels, in_channels // 2, dim=dim,
                                              share_nonlinearity=share_nonlinearity, use_batchnorm=use_batchnorm)
        self.vn2 = VNSimpleLinearAndLeakyReLU(in_channels // 2, in_channels // 4, dim=dim,
                                              share_nonlinearity=share_nonlinearity, use_batchnorm=use_batchnorm)
        self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)

        return x_std, z0


class EQCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(EQCNN_cls, self).__init__()
        self.args = args
        self.k = args.k
        self.use_batchnorm = args.use_batchnorm
        self.inv_use_mean = args.inv_use_mean

        if args.combine_lin_nonlin:
            self.conv1 = VNLinearLeakyReLU(2, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv2 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 128 // 3, use_batchnorm=args.use_batchnorm)
            self.conv4 = VNLinearLeakyReLU(128 // 3 * 2, 256 // 3, use_batchnorm=args.use_batchnorm)

            self.conv5 = VNLinearLeakyReLU(256 // 3 + 128 // 3 + 64 // 3 * 2, args.emb_dims // 3, dim=4,
                                           share_nonlinearity=True, use_batchnorm=args.use_batchnorm)
        else:
            self.conv1 = VNLinearAndLeakyReLU(2, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv2 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv3 = VNLinearAndLeakyReLU(64 // 3 * 2, 128 // 3, use_batchnorm=args.use_batchnorm)
            self.conv4 = VNLinearAndLeakyReLU(128 // 3 * 2, 256 // 3, use_batchnorm=args.use_batchnorm)

            self.conv5 = VNLinearAndLeakyReLU(256 // 3 + 128 // 3 + 64 // 3 * 2, args.emb_dims // 3, dim=4,
                                              share_nonlinearity=True, use_batchnorm=args.use_batchnorm)

        if args.inv_use_mean:
            self.std_feature = VNStdFeature(args.emb_dims // 3 * 2, dim=4, n_layers=args.inv_mlp_layers,
                                            normalize_frame=False, use_batchnorm=args.use_batchnorm,
                                            combine_lin_nonlin=args.combine_lin_nonlin)
            self.linear1 = nn.Linear((args.emb_dims // 3) * 12, 512)
        else:
            self.std_feature = VNStdFeature(args.emb_dims // 3, dim=4, n_layers=args.inv_mlp_layers,
                                            normalize_frame=False, use_batchnorm=args.use_batchnorm,
                                            combine_lin_nonlin=args.combine_lin_nonlin)
            self.linear1 = nn.Linear((args.emb_dims // 3) * 6, 512)

        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(128 // 3)
            self.pool4 = VNMaxPool(256 // 3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        num_points = x.size(-1)
        if self.inv_use_mean:
            x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
            x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class EQCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(EQCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.use_batchnorm = args.use_batchnorm
        self.inv_use_mean = args.inv_use_mean
        self.use_embedding = args.use_embedding

        self.embedding = VNEmbedding()
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        if args.use_embedding:
            init_dim = 4
        else:
            init_dim = 2

        if args.combine_lin_nonlin:
            if args.reflection_variant:
                if args.b_only:
                    self.conv1 = VNReflectionLinearLeakyReLUBOnly(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv2 = VNReflectionLinearLeakyReLUBOnly(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv3 = VNReflectionLinearLeakyReLUBOnly(64 // 3 * 2, 64 // 3,
                                                                  use_batchnorm=args.use_batchnorm)
                    self.conv4 = VNReflectionLinearLeakyReLUBOnly(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv5 = VNReflectionLinearLeakyReLUBOnly(64 // 3 * 2, 64 // 3,
                                                                  use_batchnorm=args.use_batchnorm)
                else:
                    self.conv1 = VNReflectionLinearLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv2 = VNReflectionLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv3 = VNReflectionLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv4 = VNReflectionLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv5 = VNReflectionLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            else:
                self.conv1 = VNLinearLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
        else:
            if args.reflection_variant:
                self.conv1 = VNReflectionLinearAndLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv2 = VNReflectionLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv3 = VNReflectionLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv4 = VNReflectionLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv5 = VNReflectionLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            else:
                self.conv1 = VNLinearAndLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv2 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv3 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv4 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv5 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)

        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        if args.reflection_variant:
            if args.b_only:
                self.conv6 = VNReflectionLinearLeakyReLUBOnly(64 // 3 * 3, args.emb_dims // 3, dim=4,
                                                              share_nonlinearity=True, use_batchnorm=args.use_batchnorm)
            else:
                self.conv6 = VNReflectionLinearLeakyReLU(64 // 3 * 3, args.emb_dims // 3, dim=4,
                                                         share_nonlinearity=True, use_batchnorm=args.use_batchnorm)
        else:
            self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, args.emb_dims // 3, dim=4, share_nonlinearity=True,
                                           use_batchnorm=args.use_batchnorm)
        if args.inv_use_mean:
            self.std_feature = VNStdFeature(args.emb_dims // 3 * 2, dim=4, n_layers=args.inv_mlp_layers,
                                            normalize_frame=False, use_batchnorm=args.use_batchnorm,
                                            combine_lin_nonlin=args.combine_lin_nonlin,
                                            reflection_variant=args.reflection_variant, b_only=args.b_only)
            self.conv8 = nn.Sequential(nn.Conv1d(2299, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
        else:
            self.std_feature = VNStdFeature(args.emb_dims // 3, dim=4, n_layers=args.inv_mlp_layers,
                                            normalize_frame=False, use_batchnorm=args.use_batchnorm,
                                            combine_lin_nonlin=args.combine_lin_nonlin,
                                            reflection_variant=args.reflection_variant, b_only=args.b_only)
            self.conv8 = nn.Sequential(nn.Conv1d(2299 - args.emb_dims // 3 * 3, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))

        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x_in, l):
        batch_size = x_in.size(0)
        num_points = x_in.size(2)

        x = x_in.unsqueeze(1)
        x_coord = x

        if self.use_embedding:
            x = self.embedding(x)  # (batch_size, 3, num_points) -> (batch_size, 3*init_dim, num_points, 1)
        else:
            x = get_graph_feature(x,
                                  k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*init_dim, num_points, 40)
        x = self.conv1(x)  # (batch_size, 3*4, num_points, ...) -> (batch_size, 64, num_points, ...)
        x = self.conv2(x)  # (batch_size, 64, num_points, ...) -> (batch_size, 64, num_points, ...)
        x1 = self.pool1(x)  # (batch_size, 64, num_points, ...) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = self.pool2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = self.pool3(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x123 = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x123)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)
        x = x.view(batch_size, -1, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x123), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)  # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)

        return x


def kd_mean_pool(x, pool_size):
    r = len(list(x.size()))
    if r == 4:
        x = x.view(x.size(0), x.size(1), x.size(2), -1, pool_size)
    if r == 3:
        x = x.view(x.size(0), x.size(1), -1, pool_size)
    return x.mean(dim=-1, keepdim=False)


def sampled_get_graph_feature(x, y, k=20, idx=None, x_coord_src=None, x_coord_tar=None, use_x_coord=False):
    batch_size = x.size(0)
    num_points_src = x.size(3)
    num_points_tar = y.size(3)
    x = x.view(batch_size, -1, num_points_src)
    y = y.view(batch_size, -1, num_points_tar)
    if idx is None:
        if not use_x_coord:  # dynamic knn graph
            idx = sampled_knn(x, y, k=k)  # (batch_size, num_points_tar, k)
        else:  # fixed knn graph with input point coordinates
            x_coord_src = x_coord_src.view(batch_size, -1, num_points_src)
            x_coord_tar = x_coord_tar.view(batch_size, -1, num_points_tar)
            # if we just do idx = knn(x_coord, k=k), we get nan loss
            idx = sampled_knn(x_coord_src, x_coord_tar, k=k)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points_src  # (batch_size)

    idx = idx + idx_base

    idx = idx.view(-1)  # (batch_size * num_points_tar * k)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points_src, E * 3)
    y = y.transpose(2, 1).contiguous()  # (batch_size, num_points_tar, E * 3)
    feature = x.view(batch_size * num_points_src, -1)[idx, :]  # (batch_size * num_points_tar * k, E * 3)
    feature = feature.view(batch_size, num_points_tar, k, num_dims, 3)  # (batch_size, num_points_tar, k, E, 3)
    y = y.view(batch_size, num_points_tar, 1, num_dims, 3).repeat(1, 1, k, 1,
                                                                  1)  # (batch_size, num_points_tar, k, E, 3)

    feature = torch.cat((feature - y, y), dim=3).permute(0, 3, 4, 1,
                                                         2).contiguous()  # (batch_size, num_points_tar, k, E, 3)

    return feature


def sampled_knn(x, y, k):
    # x: B x E * 3 x C_x
    # y: B x E * 3 x C_y
    # distance between (x1, x2, x3) and (y1, y2, y3) is 
    # sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2 + (x3 - y3) ** 2) = 
    # sqrt((x1 ** 2 + x2 ** 2 + x3 ** 2) + (y1 ** 2 + y2 ** 2 + y3 ** 2) -2 (x1y1 + x2y2 + x3y3)) =
    # Since we only care about ordering
    # (x1 ** 2 + x2 ** 2 + x3 ** 2) + (y1 ** 2 + y2 ** 2 + y3 ** 2) -2 (x1y1 + x2y2 + x3y3)
    
    # calculate -2 (x1y1 + x2y2 + x3y3) term
    inner = -2*torch.matmul(y.transpose(2, 1), x) # B x C_y x C_x

    # calculate (x1 ** 2 + x2 ** 2 + x3 ** 2) terms
    xx = torch.sum(x**2, dim=1, keepdim=True) # B x 1 x C_x
    # calculate (y1 ** 2 + y2 ** 2 + y3 ** 2) terms
    yy = torch.sum(y ** 2, dim=1, keepdim=True) # B x 1 x C_y

    # negates distance for topk function
    pairwise_distance = -xx - inner - yy.transpose(2, 1) # B x C_y x C_x
 
    # gets topk indices
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # B x C_y x K
    return idx


def sampled_get_graph_feature_channels(x, y, k=20, idx=None, x_coord_src=None, x_coord_tar=None, use_x_coord=False):
    batch_size = x.size(0)
    num_points_src = x.size(3)
    num_points_tar = y.size(3)
    x = x.view(batch_size, -1, num_points_src)
    y = y.view(batch_size, -1, num_points_tar)
    if idx is None:
        if not use_x_coord:  # dynamic knn graph
            idx = sampled_knn(x, y, k=k)  # (batch_size, num_points_tar, k)
        else:  # fixed knn graph with input point coordinates
            x_coord_src = x_coord_src.view(batch_size, -1, num_points_src)
            x_coord_tar = x_coord_tar.view(batch_size, -1, num_points_tar)
            # if we just do idx = knn(x_coord, k=k), we get nan loss
            idx = sampled_knn(x_coord_src, x_coord_tar, k=k)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points_src  # (batch_size)

    idx = idx + idx_base

    idx = idx.view(-1)  # (batch_size * num_points_tar * k)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points_src, E * 3)
    y = y.transpose(2, 1).contiguous()  # (batch_size, num_points_tar, E * 3)
    feature = x.view(batch_size * num_points_src, -1)[idx, :]  # (batch_size * num_points_tar * k, E * 3)
    feature = feature.view(batch_size, num_points_tar, k, num_dims, 3)  # (batch_size, num_points_tar, k, E, 3)

    points = y.view(batch_size, num_points_tar, 1, num_dims, 3).permute(0, 3, 4, 1, 2).contiguous()  # B x E x 3 x C x 1
    y = y.view(batch_size, num_points_tar, 1, num_dims, 3).repeat(1, 1, k, 1,
                                                                  1)  # (batch_size, num_points_tar, k, E, 3)
    neighbors = (feature - y).permute(0, 3, 4, 1, 2).contiguous()  # B x E x 3 x C x k

    return points, neighbors


class EQCNN_sampling_VNN(nn.Module):
    def __init__(self, args, seg_num_all):
        super(EQCNN_sampling_VNN, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.use_batchnorm = args.use_batchnorm
        self.inv_use_mean = args.inv_use_mean
        self.use_embedding = args.use_embedding

        self.embedding = VNEmbedding()
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        if args.use_embedding:
            init_dim = 4
        else:
            init_dim = 2

        if args.combine_lin_nonlin:
            if args.reflection_variant:
                if args.b_only:
                    self.conv1 = VNReflectionLinearLeakyReLUBOnly(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv2 = VNReflectionLinearLeakyReLUBOnly(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv3 = VNReflectionLinearLeakyReLUBOnly(64 // 3 * 2, 64 // 3,
                                                                  use_batchnorm=args.use_batchnorm)
                    self.conv4 = VNReflectionLinearLeakyReLUBOnly(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv5 = VNReflectionLinearLeakyReLUBOnly(64 // 3 * 2, 64 // 3,
                                                                  use_batchnorm=args.use_batchnorm)
                else:
                    self.conv1 = VNReflectionLinearLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv2 = VNReflectionLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv3 = VNReflectionLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv4 = VNReflectionLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv5 = VNReflectionLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            else:
                self.conv1 = VNLinearLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
        else:
            if args.reflection_variant:
                self.conv1 = VNReflectionLinearAndLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv2 = VNReflectionLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv3 = VNReflectionLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv4 = VNReflectionLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv5 = VNReflectionLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            else:
                self.conv1 = VNLinearAndLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv2 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv3 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv4 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv5 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)

        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        if args.reflection_variant:
            if args.b_only:
                self.conv6 = VNReflectionLinearLeakyReLUBOnly(64 // 3 * 3, args.emb_dims // 3, dim=4,
                                                              share_nonlinearity=True, use_batchnorm=args.use_batchnorm)
            else:
                self.conv6 = VNReflectionLinearLeakyReLU(64 // 3 * 3, args.emb_dims // 3, dim=4,
                                                         share_nonlinearity=True, use_batchnorm=args.use_batchnorm)
        else:
            self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, args.emb_dims // 3, dim=4, share_nonlinearity=True,
                                           use_batchnorm=args.use_batchnorm)
        if args.inv_use_mean:
            self.std_feature = VNStdFeature(args.emb_dims // 3 * 2, dim=4, n_layers=args.inv_mlp_layers,
                                            normalize_frame=False, use_batchnorm=args.use_batchnorm,
                                            combine_lin_nonlin=args.combine_lin_nonlin,
                                            reflection_variant=args.reflection_variant, b_only=args.b_only)
            self.conv8 = nn.Sequential(nn.Conv1d(2488, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
        else:
            self.std_feature = VNStdFeature(args.emb_dims // 3, dim=4, n_layers=args.inv_mlp_layers,
                                            normalize_frame=False, use_batchnorm=args.use_batchnorm,
                                            combine_lin_nonlin=args.combine_lin_nonlin,
                                            reflection_variant=args.reflection_variant, b_only=args.b_only)
            self.conv8 = nn.Sequential(nn.Conv1d(2488 - args.emb_dims // 3 * 3, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))

        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x_in, l):
        batch_size = x_in.size(0)

        x = x_in.unsqueeze(1)
        x_kd = kd_mean_pool(x, 1)  # B x 1 x 3 x C

        if self.use_embedding:
            x = self.embedding(x_kd)  # (batch_size, 3, num_points) -> (batch_size, 3*init_dim, num_points, 1)
        else:
            x = sampled_get_graph_feature(x, x_kd, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*init_dim, num_points, 40)
        x = self.conv1(x)  # (batch_size, 3*4, num_points, ...) -> (batch_size, 64, num_points, ...)
        x = self.conv2(x)  # (batch_size, 64, num_points, ...) -> (batch_size, 64, num_points, ...)
        x1 = self.pool1(x)  # (batch_size, 64, num_points, ...) -> (batch_size, 64, num_points)
        x1_kd = kd_mean_pool(x1, 2)  # B x E / 3 x 3 x C / 2

        x = sampled_get_graph_feature(x1, x1_kd,
                                      k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = self.pool2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2_kd = kd_mean_pool(x2, 2)  # B x E / 3 x 3 x C / 4

        x = sampled_get_graph_feature(x2, x2_kd,
                                      k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = self.pool3(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x1_reduced = kd_mean_pool(x1, 8)  # B x E / 3 x 3 x C / 8
        x2_reduced = kd_mean_pool(x2, 4)  # B x E / 3 x 3 x C / 8
        x3_reduced = kd_mean_pool(x3, 2)  # B x E / 3 x 3 x C / 8

        x123 = torch.cat((x1_reduced, x2_reduced, x3_reduced), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x123)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)

        x = torch.repeat_interleave(x, 2, dim=3)  # B x 2E' x 3 x C / 4
        x = torch.cat((x, x3), dim=1)  # B x 2E' + E / 3 x 3 x C / 4
        x = torch.repeat_interleave(x, 2, dim=3)  # B x 2E' + E / 3 x 3 x C / 2
        x = torch.cat((x, x2), dim=1)  # B x 2E' + 2 * E / 3 x 3 x C / 2
        x = torch.repeat_interleave(x, 2, dim=3)  # B x 2E' + 2 * E / 3 x 3 x C
        x = torch.cat((x, x1), dim=1)  # B x 2E' + E x 3 x C

        x123 = torch.repeat_interleave(x123, 8, dim=3)  # B x E x 3 x C
        z0 = torch.repeat_interleave(z0, 8, dim=3)  # 1 x 3 x 3 x C

        num_subsampled_points = x.size(3)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_subsampled_points)
        x = x.view(batch_size, -1, num_subsampled_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_subsampled_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x123), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        pred_sampled_seg = self.conv11(x)  # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)

        return pred_sampled_seg


class EQCNN_channels_VNN(nn.Module):
    def __init__(self, args, seg_num_all):
        super(EQCNN_channels_VNN, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.use_batchnorm = args.use_batchnorm
        self.inv_use_mean = args.inv_use_mean
        self.num_shells = args.num_shells

        self.embedding = VNEmbedding()
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        init_dim = 4

        if args.combine_lin_nonlin:
            if args.reflection_variant:
                if args.b_only:
                    self.conv1 = VNReflectionLinearLeakyReLUBOnly(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv2 = VNReflectionLinearLeakyReLUBOnly(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv3 = VNReflectionLinearLeakyReLUBOnly(64 // 3 * 2, 64 // 3,
                                                                  use_batchnorm=args.use_batchnorm)
                    self.conv4 = VNReflectionLinearLeakyReLUBOnly(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv5 = VNReflectionLinearLeakyReLUBOnly(64 // 3 * 2, 64 // 3,
                                                                  use_batchnorm=args.use_batchnorm)
                else:
                    self.conv1 = VNReflectionLinearLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv2 = VNReflectionLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv3 = VNReflectionLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv4 = VNReflectionLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv5 = VNReflectionLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            else:
                self.conv1 = VNLinearLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
        else:
            if args.reflection_variant:
                self.conv1 = VNReflectionLinearAndLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv2 = VNReflectionLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv3 = VNReflectionLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv4 = VNReflectionLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv5 = VNReflectionLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            else:
                self.conv1 = VNLinearAndLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv2 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv3 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv4 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv5 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)

        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        if args.reflection_variant:
            if args.b_only:
                self.conv6 = VNReflectionLinearLeakyReLUBOnly(64 // 3 * 3, args.emb_dims // 3, dim=4,
                                                              share_nonlinearity=True, use_batchnorm=args.use_batchnorm)
            else:
                self.conv6 = VNReflectionLinearLeakyReLU(64 // 3 * 3, args.emb_dims // 3, dim=4,
                                                         share_nonlinearity=True, use_batchnorm=args.use_batchnorm)
        else:
            self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, args.emb_dims // 3, dim=4, share_nonlinearity=True,
                                           use_batchnorm=args.use_batchnorm)
        if args.inv_use_mean:
            self.std_feature = VNStdFeature(args.emb_dims // 3 * 2, dim=4, n_layers=args.inv_mlp_layers,
                                            normalize_frame=False, use_batchnorm=args.use_batchnorm,
                                            combine_lin_nonlin=args.combine_lin_nonlin,
                                            reflection_variant=args.reflection_variant, b_only=args.b_only)
            self.conv8 = nn.Sequential(nn.Conv1d(2488, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
        else:
            self.std_feature = VNStdFeature(args.emb_dims // 3, dim=4, n_layers=args.inv_mlp_layers,
                                            normalize_frame=False, use_batchnorm=args.use_batchnorm,
                                            combine_lin_nonlin=args.combine_lin_nonlin,
                                            reflection_variant=args.reflection_variant, b_only=args.b_only)
            self.conv8 = nn.Sequential(nn.Conv1d(2488 - args.emb_dims // 3 * 3, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))

        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x_in, l):
        batch_size = x_in.size(0)
        num_points = x_in.size(2)

        x = x_in.unsqueeze(1)
        x_kd = kd_mean_pool(x, 1)  # B x 1 x 3 x C

        points, neighbors = sampled_get_graph_feature_channels(x, x_kd, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*init_dim, num_points, 40)
        
        rot_cross, rot_shell = orientation_embd(x_in.permute(0, 2, 1), num_points // self.num_shells, self.num_shells)  # B x C * N x 3
        B, _, _, C, _ = points.size()
        N = (self.num_shells * (self.num_shells - 1)) // 2
        rot_cross = rot_cross.view(B, 1, C, N, 3).permute(0, 1, 4, 2, 3)  # B x E=1 x 3 x C x N
        rot_shell = rot_shell.view(B, 1, C, N, 3).permute(0, 1, 4, 2, 3)  # B x E=1 x 3 x C x N

        points = points.repeat(1, 1, 1, 1, self.k * N)
        neighbors = neighbors.repeat(1, 1, 1, 1, N)
        rot_shell = rot_shell.repeat(1, 1, 1, 1, self.k)
        rot_cross = rot_cross.repeat(1, 1, 1, 1, self.k)

        x = torch.cat((points, neighbors, rot_shell, rot_cross), 1)  # B x E=4 x 3 x C x K * N

        x = self.conv1(x)  # (batch_size, 3*4, num_points, ...) -> (batch_size, 64, num_points, ...)
        x = self.conv2(x)  # (batch_size, 64, num_points, ...) -> (batch_size, 64, num_points, ...)
        x1 = self.pool1(x)  # (batch_size, 64, num_points, ...) -> (batch_size, 64, num_points)
        x1_kd = kd_mean_pool(x1, 2)  # B x E / 3 x 3 x C / 2

        x = sampled_get_graph_feature(x1, x1_kd,
                                      k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = self.pool2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2_kd = kd_mean_pool(x2, 2)  # B x E / 3 x 3 x C / 4

        x = sampled_get_graph_feature(x2, x2_kd,
                                      k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = self.pool3(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x1_reduced = kd_mean_pool(x1, 8)  # B x E / 3 x 3 x C / 8
        x2_reduced = kd_mean_pool(x2, 4)  # B x E / 3 x 3 x C / 8
        x3_reduced = kd_mean_pool(x3, 2)  # B x E / 3 x 3 x C / 8

        x123 = torch.cat((x1_reduced, x2_reduced, x3_reduced), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x123)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)

        x = torch.repeat_interleave(x, 2, dim=3)  # B x 2E' x 3 x C / 4
        x = torch.cat((x, x3), dim=1)  # B x 2E' + E / 3 x 3 x C / 4
        x = torch.repeat_interleave(x, 2, dim=3)  # B x 2E' + E / 3 x 3 x C / 2
        x = torch.cat((x, x2), dim=1)  # B x 2E' + 2 * E / 3 x 3 x C / 2
        x = torch.repeat_interleave(x, 2, dim=3)  # B x 2E' + 2 * E / 3 x 3 x C
        x = torch.cat((x, x1), dim=1)  # B x 2E' + E x 3 x C

        x123 = torch.repeat_interleave(x123, 8, dim=3)  # B x E x 3 x C
        z0 = torch.repeat_interleave(z0, 8, dim=3)  # 1 x 3 x 3 x C

        num_subsampled_points = x.size(3)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_subsampled_points)
        x = x.view(batch_size, -1, num_subsampled_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_subsampled_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x123), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        pred_sampled_seg = self.conv11(x)  # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)

        return pred_sampled_seg


class EQCNN_regress(nn.Module):
    def __init__(self, args, seg_num_all, b_only=False):
        super(EQCNN_regress, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.use_batchnorm = args.use_batchnorm
        self.inv_use_mean = args.inv_use_mean
        self.include_x_coord = args.include_x_coord
        self.use_x_coord = args.use_x_coord

        self.embedding = VNEmbedding()
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        init_dim = 2

        if args.combine_lin_nonlin:
            self.conv1 = VNLinearLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
        else:
            self.conv1 = VNLinearAndLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv2 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv3 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv4 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv5 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)

        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, args.emb_dims // 3, dim=4, share_nonlinearity=True,
                                       use_batchnorm=args.use_batchnorm)
        if args.inv_use_mean:
            self.std_feature = VNStdFeature(args.emb_dims // 3 * 2, dim=4, n_layers=args.inv_mlp_layers,
                                            normalize_frame=False, use_batchnorm=args.use_batchnorm,
                                            combine_lin_nonlin=args.combine_lin_nonlin, reflection_variant=False,
                                            b_only=b_only)
            self.conv8 = nn.Sequential(nn.Conv1d(2299, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
        else:
            self.std_feature = VNStdFeature(args.emb_dims // 3, dim=4, n_layers=args.inv_mlp_layers,
                                            normalize_frame=False, use_batchnorm=args.use_batchnorm,
                                            combine_lin_nonlin=args.combine_lin_nonlin, reflection_variant=False,
                                            b_only=b_only)
            self.conv8 = nn.Sequential(nn.Conv1d(2299 - args.emb_dims // 3 * 3, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))

        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        self.regression = VNRegression(self.seg_num_all)

    def forward(self, x_in, l, device):
        batch_size = x_in.size(0)
        num_points = x_in.size(2)

        x = x_in.unsqueeze(1)
        if self.include_x_coord:
            x_coord = x
        else:
            x_coord = None

        x = get_graph_feature(x, k=self.k, x_coord=x_coord,
                              use_x_coord=self.use_x_coord)  # (batch_size, 3, num_points) -> (batch_size, 3*init_dim, num_points, 40)
        x = self.conv1(x)  # (batch_size, 3*4, num_points, ...) -> (batch_size, 64, num_points, ...)
        x = self.conv2(x)  # (batch_size, 64, num_points, ...) -> (batch_size, 64, num_points, ...)
        x1 = self.pool1(x)  # (batch_size, 64, num_points, ...) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, x_coord=x_coord,
                              use_x_coord=self.use_x_coord)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = self.pool2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, x_coord=x_coord,
                              use_x_coord=self.use_x_coord)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = self.pool3(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x123 = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x123)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)
        x = x.view(batch_size, -1, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x123), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)

        seg_pred = self.conv11(x)  # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)
        attention_weights = F.softmax(seg_pred, dim=1)
        mu = vn_get_keypoints(attention_weights,
                              x_in)  # (batch_size, seg_num_all, num_points) -> (batch_size, seg_num_all, 3)
        r = self.regression(mu)
        return seg_pred, attention_weights, mu, x_in, r


class EQCNN_ref_frame(nn.Module):
    def __init__(self, args, seg_num_all, b_only=False):
        super(EQCNN_ref_frame, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.use_batchnorm = args.use_batchnorm
        self.inv_use_mean = args.inv_use_mean
        self.include_x_coord = args.include_x_coord
        self.use_x_coord = args.use_x_coord
        self.emb_dims = args.emb_dims

        self.embedding = VNEmbedding()
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        init_dim = 2

        if args.combine_lin_nonlin:
            self.conv1 = VNLinearLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
        else:
            self.conv1 = VNLinearAndLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv2 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv3 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv4 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
            self.conv5 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)

        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, args.emb_dims // 3, dim=4, share_nonlinearity=True,
                                       use_batchnorm=args.use_batchnorm)
        self.conv7 = VNLinear(args.emb_dims // 3, 2)

    def forward(self, x_in, l, device):
        batch_size = x_in.size(0)
        num_points = x_in.size(2)

        x = x_in.unsqueeze(1)
        if self.include_x_coord:
            x_coord = x
        else:
            x_coord = None

        x = get_graph_feature(x, k=self.k, x_coord=x_coord,
                              use_x_coord=self.use_x_coord)  # (batch_size, 3, num_points) -> (batch_size, 3*init_dim, num_points, 40)
        x = self.conv1(x)  # (batch_size, 3*4, num_points, ...) -> (batch_size, 64, num_points, ...)
        x = self.conv2(x)  # (batch_size, 64, num_points, ...) -> (batch_size, 64, num_points, ...)
        x1 = self.pool1(x)  # (batch_size, 64, num_points, ...) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, x_coord=x_coord,
                              use_x_coord=self.use_x_coord)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = self.pool2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, x_coord=x_coord,
                              use_x_coord=self.use_x_coord)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = self.pool3(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x123 = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x123)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        xy = torch.mean(self.conv7(x), 3)  # (batch_size, 3, num_points) -> (batch_size, 2, 3)
        z = torch.unsqueeze(torch.cross(xy[:, 0, :], xy[:, 1, :], dim=1), 1)  # B x 1 x 3
        r = torch.cat((xy, z), dim=1)  # B x 3 x 3
        return r


class EQCNN_partseg_shell_channels(nn.Module):
    def __init__(self, args, seg_num_all):
        super(EQCNN_partseg_shell_channels, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.use_batchnorm = args.use_batchnorm
        self.inv_use_mean = args.inv_use_mean
        self.use_embedding = args.use_embedding
        self.num_points = args.num_points
        self.num_shells = args.num_shells

        self.vnn_lin_1 = VNSimpleLinear(3, args.emb_dims//3)
        self.embedding = VNEmbedding()
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        if args.use_embedding:
            init_dim = 4
        else:
            init_dim = 344

        if args.combine_lin_nonlin:
            if args.reflection_variant:
                if args.b_only:
                    self.conv1 = VNReflectionLinearLeakyReLUBOnly(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv2 = VNReflectionLinearLeakyReLUBOnly(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv3 = VNReflectionLinearLeakyReLUBOnly(64 // 3 * 2, 64 // 3,
                                                                  use_batchnorm=args.use_batchnorm)
                    self.conv4 = VNReflectionLinearLeakyReLUBOnly(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv5 = VNReflectionLinearLeakyReLUBOnly(64 // 3 * 2, 64 // 3,
                                                                  use_batchnorm=args.use_batchnorm)
                else:
                    self.conv1 = VNReflectionLinearLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv2 = VNReflectionLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv3 = VNReflectionLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv4 = VNReflectionLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                    self.conv5 = VNReflectionLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            else:
                self.conv1 = VNLinearLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
        else:
            if args.reflection_variant:
                self.conv1 = VNReflectionLinearAndLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv2 = VNReflectionLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv3 = VNReflectionLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv4 = VNReflectionLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv5 = VNReflectionLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
            else:
                self.conv1 = VNLinearAndLeakyReLU(init_dim, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv2 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv3 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv4 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, use_batchnorm=args.use_batchnorm)
                self.conv5 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, use_batchnorm=args.use_batchnorm)

        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        if args.reflection_variant:
            if args.b_only:
                self.conv6 = VNReflectionLinearLeakyReLUBOnly(64 // 3 * 3, args.emb_dims // 3, dim=4,
                                                              share_nonlinearity=True, use_batchnorm=args.use_batchnorm)
            else:
                self.conv6 = VNReflectionLinearLeakyReLU(64 // 3 * 3, args.emb_dims // 3, dim=4,
                                                         share_nonlinearity=True, use_batchnorm=args.use_batchnorm)
        else:
            self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, args.emb_dims // 3, dim=4, share_nonlinearity=True,
                                           use_batchnorm=args.use_batchnorm)
        if args.inv_use_mean:
            self.std_feature = VNStdFeature(args.emb_dims // 3 * 2, dim=4, n_layers=args.inv_mlp_layers,
                                            normalize_frame=False, use_batchnorm=args.use_batchnorm,
                                            combine_lin_nonlin=args.combine_lin_nonlin,
                                            reflection_variant=args.reflection_variant, b_only=args.b_only)
            self.conv8 = nn.Sequential(nn.Conv1d(2299, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
        else:
            self.std_feature = VNStdFeature(args.emb_dims // 3, dim=4, n_layers=args.inv_mlp_layers,
                                            normalize_frame=False, use_batchnorm=args.use_batchnorm,
                                            combine_lin_nonlin=args.combine_lin_nonlin,
                                            reflection_variant=args.reflection_variant, b_only=args.b_only)
            self.conv8 = nn.Sequential(nn.Conv1d(2299 - args.emb_dims // 3 * 3, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))

        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x_in, l):
        batch_size = x_in.size(0)
        num_points = x_in.size(2)

        x = x_in.unsqueeze(1)
        x_coord = x

        x_graph = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*init_dim, num_points, 40)
        rot_cross, rot_embed = orientation_embd(x_in.permute(0, 2, 1), self.num_points // self.num_shells, self.num_shells) # B x C x 3
        rot_cross = torch.sum(rot_cross, dim=1) # B x 3
        rot_cross = rot_cross.unsqueeze(1).unsqueeze(-1).unsqueeze(-1) # B x 1 x 3 x 1 x 1
        rot_cross = rot_cross.repeat(1, 1, 1, self.num_points, self.k) # B x 1 x 3 x C x K
        x_graph = torch.cat((x_graph, rot_cross), dim=1).contiguous() # B x E=3 x 3 x C x K

        x = self.vnn_lin_1(x_graph)                              
        x = self.pool1(x).permute(0, 3, 1, 2) # B x C x E x 3
        K = torch.unsqueeze(x, 2) # B x C x H x E x 3
        Q = torch.unsqueeze(x, 2) # B x C x H x E x 3
        V = torch.unsqueeze(x, 2) # B x C x H x E x 3
        att = torch.squeeze(attention_layer(Q, K, V, 'cross'), 2).permute(0, 2, 3, 1) # B x E x 3 x C
        att = att.unsqueeze(-1).repeat(1, 1, 1, 1, self.k) # B x 1 x 3 x C x K
        x = torch.cat((x_graph, att), dim=1).contiguous() # B x E=3 x 3 x C x K

        x = self.conv1(x)  # (batch_size, 3*4, num_points, ...) -> (batch_size, 64, num_points, ...)
        x = self.conv2(x)  # (batch_size, 64, num_points, ...) -> (batch_size, 64, num_points, ...)
        x1 = self.pool1(x)  # (batch_size, 64, num_points, ...) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = self.pool2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = self.pool3(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x123 = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x123)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)
        x = x.view(batch_size, -1, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x123), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)  # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)

        return x


class EQCNN_complex(nn.Module):
    def __init__(self, args, seg_num_all, b_only=False):
        super(EQCNN_complex, self).__init__()
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.use_x_coord = args.use_x_coord
        self.emb_dims = args.emb_dims
        self.vnn_lin_1 = VNSimpleLinear(2, args.emb_dims // 3)
        self.vnn_lin_2 = VNSimpleLinear(2, args.emb_dims // 3)

        self.pool1 = mean_pool
        self.pool2 = mean_pool

        self.vnn_lin_3 = VNSimpleLinear(4 * (args.emb_dims // 3), args.emb_dims // 3)
        self.vnn_lin_4 = VNSimpleLinear(4 * (args.emb_dims // 3), args.emb_dims // 3)

        self.pool3 = mean_pool
        self.pool4 = mean_pool

        self.complex_lin_1 = ComplexLinearAndLeakyReLU(args.emb_dims // 3, args.emb_dims // 3)
        self.complex_lin_2 = ComplexLinearAndLeakyReLU(args.emb_dims // 3, args.emb_dims // 3)
        self.std_feature = VNSimpleStdFeature(args.emb_dims // 3 * 2, dim=4, use_batchnorm=args.use_batchnorm,
                                              combine_lin_nonlin=args.combine_lin_nonlin, reflection_variant=False,
                                              b_only=b_only)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv1d(args.emb_dims // 3 * 6, 256, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x_in, l):
        batch_size = x_in.size(0)
        num_points = x_in.size(2)
        device = torch.device("cuda")

        x = x_in.unsqueeze(1)
        x_coord = x  # B x E x 3 x C

        x_graph = get_graph_feature(x, k=self.k, x_coord=x_coord, use_x_coord=self.use_x_coord)

        x = self.vnn_lin_1(x_graph)
        x = self.pool1(x).permute(0, 3, 1, 2)  # B x C x E x 3

        j = self.vnn_lin_2(x_graph)
        j = self.pool2(j).permute(0, 3, 1, 2)  # B x C x E x 3

        y = self.complex_lin_1(x, j, device)  # B x E x 3 x C

        K = torch.unsqueeze(x, 2)  # B x C x H x E x 3
        Q = torch.unsqueeze(x, 2)  # B x C x H x E x 3
        V = torch.unsqueeze(x, 2)  # B x C x H x E x 3
        att = torch.squeeze(attention_layer(Q, K, V, 'cross'), 2).permute(0, 2, 3, 1)  # B x E x 3 x C

        combd = torch.cat((y, att), 1)  # B x 2E x 3 x C

        x_graph_2 = get_graph_feature(combd, k=self.k, x_coord=x_coord, use_x_coord=self.use_x_coord)

        x_2 = self.vnn_lin_3(x_graph_2)
        x_2 = self.pool3(x_2).permute(0, 3, 1, 2)  # B x C x E x 3

        j_2 = self.vnn_lin_4(x_graph_2)
        j_2 = self.pool4(j_2).permute(0, 3, 1, 2)  # B x C x E x 3

        y_2 = self.complex_lin_2(x_2, j_2, device)  # B x E x 3 x C

        y_mean = y_2.mean(dim=-1, keepdim=True).expand(y_2.size())  # B x E x 3 x C

        x = torch.cat((y_2, y_mean), 1)  # B x 2E x 3 x C
        x, z0 = self.std_feature(x)  # x: B x 2E x 3 x C
        # z0: 1 x 3 x 3 x C

        x = x.view(batch_size, -1, num_points)  # B x 2E * 3 x C
        x = self.conv1(x)  # B x 256 x C
        x = self.dp1(x)  # B x 256 x C
        x = self.conv2(x)  # B x 256 x C
        x = self.dp2(x)  # B x 256 x C
        x = self.conv3(x)  # B x 128 x C
        x = self.conv4(x)  # B x seg_num_all x C
        return x


def sq_dist_mat(source):
    # source = target = B x C x 3
    r0 = source * source  # B x C x 3
    r0 = torch.sum(r0, dim=2, keepdim=True)  # B x C x 1
    r1 = r0.permute(0, 2, 1)  # B x 1 x C
    sq_distance_mat = r0 - 2. * torch.matmul(source, source.permute(0, 2, 1)) + r1  # B x C x C

    return sq_distance_mat


def compute_patches(source, sq_distance_mat, num_samples):
    # source = target = B x C x 3
    # sq_distance_mat = B x C x C
    batch_size = source.size()[0]
    num_points_source = source.size()[1]
    assert (num_samples <= num_points_source)

    sq_patches_dist, patches_idx = torch.topk(-sq_distance_mat, k=num_samples, dim=-1)  # B x C x k
    sq_patches_dist = -sq_patches_dist

    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points_source
    patches_idx = patches_idx + idx_base
    patches_idx = patches_idx.view(-1)
    feature = source.view(batch_size * num_points_source, -1)[patches_idx, :]
    feature = feature.view(batch_size, num_points_source, num_samples, 3)
    return feature


def orientation_embd(x, patch_size, num_shells):
    sq_distance_mat = sq_dist_mat(x)
    patches_ = compute_patches(x, sq_distance_mat, patch_size)  # B x C x K x 3
    all_embd = torch.reshape(patches_, (
    patches_.size()[0], patches_.size()[1], num_shells, -1, 3))  # B x C x num_shells x K / num_shells x 3
    all_embd = all_embd - torch.unsqueeze(torch.unsqueeze(x, dim=-2), dim=-2)
    embd = torch.mean(all_embd, dim=-2)  # B x C x num_shells x 3

    I = []
    J = []
    for i in range(num_shells):
        for j in range(num_shells):
            if i < j:
                I.append(i)
                J.append(j)
    I = torch.Tensor(I).type(torch.int64)  # N = num_shells * (num_shells - 1) / 2
    J = torch.Tensor(J).type(torch.int64)  # N = num_shells * (num_shells - 1) / 2

    embd_I = embd[:, :, I, :]  # B x C x N x 3
    embd_J = embd[:, :, J, :]  # B x C x N x 3

    cross = torch.cross(embd_I, embd_J, dim=-1)  # B x C x N x 3
    B, C, N, _ = cross.size()
    cross = torch.reshape(cross, (B, -1, 3))
    embd = torch.reshape(cross, (B, -1, 3))
    return cross, embd


class EQCNN_complex_shell(nn.Module):
    def __init__(self, args, seg_num_all):
        super(EQCNN_complex_shell, self).__init__()
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.num_points = args.num_points
        self.num_shells = args.num_shells
        self.use_x_coord = args.use_x_coord
        self.emb_dims = args.emb_dims
        self.vnn_lin_1 = VNSimpleLinear(3, args.emb_dims//3)
        self.vnn_lin_2 = VNSimpleLinear(3, args.emb_dims//3)   
        
        self.pool1 = mean_pool
        self.pool2 = mean_pool  

        self.vnn_lin_3 = VNSimpleLinear(4 * (args.emb_dims//3), args.emb_dims//3)
        self.vnn_lin_4 = VNSimpleLinear(4 * (args.emb_dims//3), args.emb_dims//3)   
        
        self.pool3 = mean_pool
        self.pool4 = mean_pool  

        self.complex_lin_1 = ComplexLinearAndLeakyReLU(args.emb_dims//3, args.emb_dims//3)
        self.complex_lin_2 = ComplexLinearAndLeakyReLU(args.emb_dims//3, args.emb_dims//3)
        self.std_feature = VNSimpleStdFeature(args.emb_dims//3*2, dim=4, use_batchnorm=args.use_batchnorm, 
            combine_lin_nonlin=args.combine_lin_nonlin, reflection_variant=False)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv1d(args.emb_dims//3*6, 256, kernel_size=1, bias=False),
                               self.bn2,
                               nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x_in, l):
        batch_size = x_in.size(0)
        num_points = x_in.size(2)
        device = torch.device("cuda")
        
        x = x_in.unsqueeze(1)
        x_coord = x #B x E x 3 x C

        x_graph = get_graph_feature(x, k=self.k, x_coord=x_coord, use_x_coord=self.use_x_coord) # B x E=2 x 3 x C x K
        
        rot_cross, rot_embed = orientation_embd(x_in.permute(0, 2, 1), self.num_points // self.num_shells, self.num_shells) # B x C x 3
        rot_cross = torch.sum(rot_cross, dim=1) # B x 3
        rot_cross = rot_cross.unsqueeze(1).unsqueeze(-1).unsqueeze(-1) # B x 1 x 3 x 1 x 1
        rot_cross = rot_cross.repeat(1, 1, 1, self.num_points, self.k) # B x 1 x 3 x C x K
        x_graph = torch.cat((x_graph, rot_cross), dim=1).contiguous() # B x E=3 x 3 x C x K

        x = self.vnn_lin_1(x_graph)                              
        x = self.pool1(x).permute(0, 3, 1, 2) # B x C x E x 3

        j = self.vnn_lin_2(x_graph)                           
        j = self.pool2(j).permute(0, 3, 1, 2) # B x C x E x 3  

        y = self.complex_lin_1(x, j, device) # B x E x 3 x C
        
        K = torch.unsqueeze(x, 2) # B x C x H x E x 3
        Q = torch.unsqueeze(x, 2) # B x C x H x E x 3
        V = torch.unsqueeze(x, 2) # B x C x H x E x 3
        att = torch.squeeze(attention_layer(Q, K, V, 'cross'), 2).permute(0, 2, 3, 1) # B x E x 3 x C

        combd = torch.cat((y, att), 1) # B x 2E x 3 x C
      
        x_graph_2 = get_graph_feature(combd, k=self.k, x_coord=x_coord, use_x_coord=self.use_x_coord) 

        x_2 = self.vnn_lin_3(x_graph_2)                              
        x_2 = self.pool3(x_2).permute(0, 3, 1, 2) # B x C x E x 3

        j_2 = self.vnn_lin_4(x_graph_2)                           
        j_2 = self.pool4(j_2).permute(0, 3, 1, 2) # B x C x E x 3 

        y_2 = self.complex_lin_2(x_2, j_2, device) # B x E x 3 x C
        
        y_mean = y_2.mean(dim=-1, keepdim=True).expand(y_2.size()) # B x E x 3 x C

        x = torch.cat((y_2, y_mean), 1) # B x 2E x 3 x C
        x, z0 = self.std_feature(x) # x: B x 2E x 3 x C
                                    # z0: 1 x 3 x 3 x C

        x = x.view(batch_size, -1, num_points) # B x 2E * 3 x C
        x = self.conv1(x) # B x 256 x C
        x = self.dp1(x) # B x 256 x C
        x = self.conv2(x) # B x 256 x C
        x = self.dp2(x) # B x 256 x C
        x = self.conv3(x) # B x 128 x C
        x = self.conv4(x) # B x seg_num_all x C
        return x


class EQCNN_complex_shell_single_complex(nn.Module):
    def __init__(self, args, seg_num_all):
        super(EQCNN_complex_shell_single_complex, self).__init__()
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.num_points = args.num_points
        self.num_shells = args.num_shells
        self.use_x_coord = args.use_x_coord
        self.emb_dims = args.emb_dims
        self.vnn_lin_1 = VNSimpleLinear(4, args.emb_dims // 3)
        self.vnn_lin_2 = VNSimpleLinear(4, args.emb_dims // 3)

        self.pool1 = mean_pool
        self.pool2 = mean_pool

        self.complex_lin_1 = ComplexLinearAndLeakyReLU(args.emb_dims // 3, args.emb_dims // 3)
        self.std_feature = VNSimpleStdFeature(args.emb_dims // 3 * 2, dim=4, use_batchnorm=args.use_batchnorm,
                                              combine_lin_nonlin=args.combine_lin_nonlin, reflection_variant=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv1d(args.emb_dims // 3 * 6, 256, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x_in, l):
        batch_size = x_in.size(0)
        num_points = x_in.size(2)
        device = torch.device("cuda")

        x = x_in.unsqueeze(1)
        x_coord = x  # B x E x 3 x C

        x_graph = get_graph_feature(x, k=self.k, x_coord=x_coord, use_x_coord=self.use_x_coord)  # B x E=2 x 3 x C x K

        rot_cross, rot_embed = orientation_embd(x_in.permute(0, 2, 1), self.num_points // self.num_shells,
                                                self.num_shells)  # B x C x 3
        rot_cross = torch.sum(rot_cross, dim=1)  # B x 3
        rot_cross = rot_cross.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # B x 1 x 3 x 1 x 1
        rot_cross = rot_cross.repeat(1, 1, 1, self.num_points, self.k)  # B x 1 x 3 x C x K

        x = x.permute(0, 3, 1, 2)  # B x C x E x 3
        K = torch.unsqueeze(x, 2)  # B x C x H x E x 3
        Q = torch.unsqueeze(x, 2)  # B x C x H x E x 3
        V = torch.unsqueeze(x, 2)  # B x C x H x E x 3
        att = torch.squeeze(attention_layer(Q, K, V, 'cross'), 2).permute(0, 2, 3, 1)  # B x E=1 x 3 x C
        att = att.unsqueeze(4).repeat(1, 1, 1, 1, self.k)  # B x E=1 x 3 x C x K

        x_graph = torch.cat((x_graph, rot_cross, att), dim=1).contiguous()  # B x E=4 x 3 x C x K

        x = self.vnn_lin_1(x_graph)
        x = self.pool1(x).permute(0, 3, 1, 2)  # B x C x E x 3

        j = self.vnn_lin_2(x_graph)
        j = self.pool2(j).permute(0, 3, 1, 2)  # B x C x E x 3

        y = self.complex_lin_1(x, j, device)  # B x E x 3 x C

        y_mean = y.mean(dim=-1, keepdim=True).expand(y.size())  # B x E x 3 x C

        x = torch.cat((y, y_mean), 1)  # B x 2E x 3 x C
        x, z0 = self.std_feature(x)  # x: B x 2E x 3 x C
        # z0: 1 x 3 x 3 x C

        x = x.view(batch_size, -1, num_points)  # B x 2E * 3 x C
        x = self.conv1(x)  # B x 256 x C
        x = self.dp1(x)  # B x 256 x C
        x = self.conv2(x)  # B x 256 x C
        x = self.dp2(x)  # B x 256 x C
        x = self.conv3(x)  # B x 128 x C
        x = self.conv4(x)  # B x seg_num_all x C
        return x


class EQCNN_sampling_complex_shell_unet(nn.Module):
    def __init__(self, args, seg_num_all):
        super(EQCNN_sampling_complex_shell_unet, self).__init__()
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.num_points = args.num_points
        self.num_shells = args.num_shells
        self.use_x_coord = args.use_x_coord
        self.emb_dims = args.emb_dims
        self.vnn_lin_1 = VNSimpleLinear(6, args.emb_dims // 3)
        self.vnn_lin_2 = VNSimpleLinear(6, args.emb_dims // 3)

        self.pool1 = mean_pool
        self.pool2 = mean_pool

        self.complex_lin_1 = ComplexLinearAndLeakyReLU(args.emb_dims // 3, args.emb_dims // 3)
        self.std_feature = VNSimpleStdFeature(args.emb_dims // 3 * 2, dim=4, use_batchnorm=args.use_batchnorm,
                                              combine_lin_nonlin=args.combine_lin_nonlin, reflection_variant=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv1d(args.emb_dims // 3 * 12, 256, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x_in, l):
        batch_size = x_in.size(0)
        num_points = x_in.size(2)
        device = torch.device("cuda")

        x = x_in.unsqueeze(1)  # B x 1 x 3 x C
        x_kd = kd_mean_pool(x, 1)  # B x 1 x 3 x C

        points, neighbors = sampled_get_graph_feature_channels(x, x_kd, k=self.k)  # B x E=1 x 3 x C x 1
        # B x E=1 x 3 x C x K

        rot_cross, rot_shell = orientation_embd(x_in.permute(0, 2, 1), num_points // self.num_shells,
                                                self.num_shells)  # B x C * N x 3
        avg_cross = torch.sum(rot_cross, dim=1)  # B x 3
        avg_cross = avg_cross.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # B x 1 x 3 x 1 x 1
        avg_cross = avg_cross.repeat(1, 1, 1, num_points, 1)  # B x E=1 x 3 x C x 1
        B, _, _, C, _ = avg_cross.size()
        N = (self.num_shells * (self.num_shells - 1)) // 2
        rot_cross = rot_cross.view(B, 1, C, N, 3).permute(0, 1, 4, 2, 3)  # B x E=1 x 3 x C x N
        rot_shell = rot_shell.view(B, 1, C, N, 3).permute(0, 1, 4, 2, 3)  # B x E=1 x 3 x C x N

        x = x.permute(0, 3, 1, 2)  # B x C x E x 3
        K = torch.unsqueeze(x, 2)  # B x C x H x E x 3
        Q = torch.unsqueeze(x, 2)  # B x C x H x E x 3
        V = torch.unsqueeze(x, 2)  # B x C x H x E x 3
        att = torch.squeeze(attention_layer(Q, K, V, 'cross'), 2).permute(0, 2, 3, 1).unsqueeze(
            4)  # B x E=1 x 3 x C x 1

        points = points.repeat(1, 1, 1, 1, self.k * N)
        neighbors = neighbors.repeat(1, 1, 1, 1, N)
        avg_cross = avg_cross.repeat(1, 1, 1, 1, self.k * N)
        rot_shell = rot_shell.repeat(1, 1, 1, 1, self.k)
        rot_cross = rot_cross.repeat(1, 1, 1, 1, self.k)
        att = att.repeat(1, 1, 1, 1, self.k * N)

        x_graph = torch.cat((points, neighbors, avg_cross, rot_shell, rot_cross, att), 1)  # B x E=6 x 3 x C x K * N

        x = self.vnn_lin_1(x_graph)
        x_pool = self.pool1(x)  # B x E x 3 x C
        x_kd = kd_mean_pool(x_pool, 8).permute(0, 3, 1, 2)  # B x C / 8 x E x 3

        j = self.vnn_lin_2(x_graph)
        j = self.pool2(j)  # B x E x 3 x C
        j_kd = kd_mean_pool(j, 8).permute(0, 3, 1, 2)  # B x C / 8 x E x 3

        y = self.complex_lin_1(x_kd, j_kd, device)  # B x E x 3 x C / 8

        y_mean = y.mean(dim=-1, keepdim=True).expand(y.size())  # B x E x 3 x C / 8

        x = torch.cat((y, y_mean), 1)  # B x 2E x 3 x C / 2
        x, z0 = self.std_feature(x)  # x: B x 2E x 3 x C / 2
        # z0: 1 x 3 x 3 x C / 2

        x = torch.repeat_interleave(x, 8, dim=3)  # B x 2E x 3 x C
        x = torch.cat((x, x_pool, j), dim=1)  # B x 4E x 3 x C

        num_subsampled_points = x.size(3)
        x = x.view(batch_size, -1, num_subsampled_points)  # B x 2E * 3 x C
        x = self.conv1(x)  # B x 256 x C
        x = self.dp1(x)  # B x 256 x C
        x = self.conv2(x)  # B x 256 x C
        x = self.dp2(x)  # B x 256 x C
        x = self.conv3(x)  # B x 128 x C
        x = self.conv4(x)  # B x seg_num_all x C

        return x


class EQCNN_complex_shell_dotinvar(nn.Module):
    def __init__(self, args, seg_num_all, b_only=False):
        super(EQCNN_complex_shell_dotinvar, self).__init__()
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.num_points = args.num_points
        self.num_shells = args.num_shells
        self.use_x_coord = args.use_x_coord
        self.emb_dims = args.emb_dims
        self.vnn_lin_1 = VNSimpleLinear(3, args.emb_dims // 3)
        self.vnn_lin_2 = VNSimpleLinear(3, args.emb_dims // 3)

        self.pool1 = mean_pool
        self.pool2 = mean_pool

        self.vnn_lin_3 = VNSimpleLinear(4 * (args.emb_dims // 3), args.emb_dims // 3)
        self.vnn_lin_4 = VNSimpleLinear(4 * (args.emb_dims // 3), args.emb_dims // 3)

        self.pool3 = mean_pool
        self.pool4 = mean_pool

        self.complex_lin_1 = ComplexLinearAndLeakyReLU(args.emb_dims // 3, args.emb_dims // 3)
        self.complex_lin_2 = ComplexLinearAndLeakyReLU(args.emb_dims // 3, args.emb_dims // 3)
        self.std_feature = VNSimpleStdFeature(args.emb_dims // 3 * 2, dim=4, use_batchnorm=args.use_batchnorm,
                                              combine_lin_nonlin=args.combine_lin_nonlin, reflection_variant=False,
                                              b_only=b_only)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv1d(args.emb_dims // 3, 256, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x_in, l, device):
        batch_size = x_in.size(0)
        num_points = x_in.size(2)

        x = x_in.unsqueeze(1)
        x_coord = x  # B x E x 3 x C

        x_graph = get_graph_feature(x, k=self.k, x_coord=x_coord, use_x_coord=self.use_x_coord)  # B x E=2 x 3 x C x K

        rot_cross, rot_embed = orientation_embd(x_in.permute(0, 2, 1), num_points // self.num_shells,
                                                self.num_shells)  # B x C x 3
        rot_cross = torch.sum(rot_cross, dim=1)  # B x 3
        rot_cross = rot_cross.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # B x 1 x 3 x 1 x 1
        rot_cross = rot_cross.repeat(1, 1, 1, num_points, self.k)  # B x 1 x 3 x C x K
        x_graph = torch.cat((x_graph, rot_cross), dim=1).contiguous()  # B x E=3 x 3 x C x K

        x = self.vnn_lin_1(x_graph)
        x = self.pool1(x).permute(0, 3, 1, 2)  # B x C x E x 3

        j = self.vnn_lin_2(x_graph)
        j = self.pool2(j).permute(0, 3, 1, 2)  # B x C x E x 3

        y = self.complex_lin_1(x, j, device)  # B x E x 3 x C

        K = torch.unsqueeze(x, 2)  # B x C x H x E x 3
        Q = torch.unsqueeze(x, 2)  # B x C x H x E x 3
        V = torch.unsqueeze(x, 2)  # B x C x H x E x 3
        att = torch.squeeze(attention_layer(Q, K, V, 'cross'), 2).permute(0, 2, 3, 1)  # B x E x 3 x C

        combd = torch.cat((y, att), 1)  # B x 2E x 3 x C

        x_graph_2 = get_graph_feature(combd, k=self.k, x_coord=x_coord, use_x_coord=self.use_x_coord)

        x_2 = self.vnn_lin_3(x_graph_2)
        x_2 = self.pool3(x_2).permute(0, 3, 1, 2)  # B x C x E x 3

        j_2 = self.vnn_lin_4(x_graph_2)
        j_2 = self.pool4(j_2).permute(0, 3, 1, 2)  # B x C x E x 3

        y_2 = self.complex_lin_2(x_2, j_2, device)  # B x E x 3 x C
        y_2 = y_2.permute(0, 3, 1, 2).unsqueeze(4)  # B x C x E x 3 x 1
        x_in = x_in.unsqueeze(1).permute(0, 3, 1, 2).unsqueeze(3)  # B x C x 1 x 1 x 3
        x = torch.matmul(x_in, y_2)  # B x C x E x 1 x 1
        x = x.squeeze(4).squeeze(3)  # B x C x E

        x = x.permute(0, 2, 1)  # B x E x C
        x = self.conv1(x)  # B x 256 x C
        x = self.dp1(x)  # B x 256 x C
        x = self.conv2(x)  # B x 256 x C
        x = self.dp2(x)  # B x 256 x C
        x = self.conv3(x)  # B x 128 x C
        x = self.conv4(x)  # B x seg_num_all x C
        return x


class EQCNN_sym_detect(nn.Module):
    def __init__(self, args):
        super(EQCNN_sym_detect, self).__init__()
        self.k = args.k
        self.use_x_coord = args.use_x_coord
        self.emb_dims = args.emb_dims
        self.vnn_lin_1 = VNSimpleLinear(1, args.emb_dims // 3)
        self.device = torch.device("cuda" if args.cuda else "cpu")

    def forward(self, x_in):
        batch_size = x_in.size(0)
        num_points = x_in.size(1)

        x = x_in.unsqueeze(1)
        x_coord = x

        x = self.vnn_lin_1(x)

        x = x.reshape(batch_size, -1, 3)
        
        unsqueezed_data = x.unsqueeze(dim=3)
        covar = torch.sum(torch.matmul(unsqueezed_data, unsqueezed_data.permute(0, 1, 3, 2)), dim=1) / (num_points - 1)
        eigenvalues, eigenvectors = np.linalg.eig(covar.cpu().detach().numpy())
        eigenvectors = torch.tensor(eigenvectors)
        eigenvalues = torch.tensor(eigenvalues)
        idx = torch.argmin(eigenvalues, dim=1)
        idx_base = torch.arange(0, batch_size) * 3
        idx = idx + idx_base
        lr_vectors = eigenvectors.reshape(batch_size*3,3)[idx, :].abs()
        repeat_vectors = lr_vectors.unsqueeze(1).repeat(1, num_points, 1).to(self.device)
        labels = torch.matmul(x_in.unsqueeze(2), repeat_vectors.unsqueeze(3)).squeeze(3).squeeze(2)
        labels = (labels > 0).long()
        return x, eigenvectors, eigenvalues, labels


def torch_fibonnacci_sphere_sampling(num_pts): # produces equispaced points on a sphere
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    S2 = np.stack([x, y, z], axis=-1)
    return torch.tensor(S2)


def select_sym(V, n_samples, n_it, h):
    S = torch_fibonnacci_sphere_sampling(num_pts=n_samples)
    
    V1 = tf.expand_dims(V, axis=-2)
    V2 = tf.expand_dims(V, axis=-3)

    C = tf.subtract(V2, V1)
    C = tf.multiply(C, C)
    C = tf.sqrt(tf.reduce_sum(C, axis=-1, keepdims=False))

    S1 = tf.expand_dims(S, axis=-2)
    S2 = tf.expand_dims(S, axis=-3)
    X = tf.subtract(S2, S1)

    X = tf.expand_dims(X, axis=0)
    X = tf.tile(X, (C.shape[0], 1, 1, 1))

    D = tf.sqrt(tf.reduce_sum(X*X, axis=-1, keepdims=False))
    K = sinkhorn(C, n_it, h)

    print(K[0, ...])

    KD = tf.multiply(K, D)


    KD = tf.reshape(KD, [KD.shape[0], -1])
    X = tf.reshape(X, [X.shape[0], -1, 3])

    idx = tf.argmax(KD, axis=-1)
    batch_idx = tf.range(0, idx.shape[0], dtype=tf.int64)
    batch_idx = tf.expand_dims(batch_idx, axis=-1)
    idx = tf.expand_dims(idx, axis=-1)
    idx = tf.stack([batch_idx, idx], axis=-1)
    X = tf.gather_nd(X, idx)
    # X = X[:, 0, :]
    X = tf.linalg.l2_normalize(X, axis=-1)
    return X


def sinkhorn(C, n_it):
    a = torch.ones((C.size()[0], C.size()[-2])) / float(C.size()[-2]) # 1 / num_points
    b = torch.ones((C.size()[0], C.size()[-1])) / float(C.size()[-2]) # 1 / num_points

    u = torch.ones((C.size()[0], C.size()[-2]))
    C_max = torch.amax(C, (-2, -1), keepdim=True)
    C = C / C_max
    K = (-C).exp()
    # K = tf.exp(-C/l)


    for i in range(n_it):
        t = torch.einsum("bij,bi->bj", K, u)
        v = b / t
        t = torch.einsum("bij,bj->bi", K, v)
        u = a / t

    Y = K * v.unsqueeze(-2) 
    Y = Y * u.unsqueeze(-1)
    return Y


class EQCNN_sinkhorn(nn.Module):
    def __init__(self, args):
        super(EQCNN_sinkhorn, self).__init__()
        self.k = args.k
        self.use_x_coord = args.use_x_coord
        self.emb_dims = args.emb_dims
        self.vnn_lin_1 = VNSimpleLinear(1, args.emb_dims // 3)
        self.device = torch.device("cuda" if args.cuda else "cpu")

    def forward(self, x_in):
        batch_size = x_in.size(0)
        num_points = x_in.size(1)

        x = x_in.unsqueeze(1)
        x_coord = x

        x = self.vnn_lin_1(x)

        x = x.reshape(batch_size, -1, 3)
        
        
        return x


class DGCNN_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x