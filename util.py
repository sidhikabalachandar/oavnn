#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
import pickle

EPS = 1e-6

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def unit_norm_loss(vectors):
    # vectors: [B, C, 3]
    norm = torch.linalg.norm(vectors, dim=2) # B x C
    output = torch.mean(torch.abs(1 - norm))
    return output


def angle_repulsion_loss(vectors):
    # vectors: [B, C, 3]
    vectors = vectors[:, :5, :]
    num = torch.matmul(vectors, torch.transpose(vectors, 1, 2)) # B x C x C
    norm = torch.linalg.norm(vectors, dim=2) # B x C
    denom = torch.matmul(torch.unsqueeze(norm, 2), torch.unsqueeze(norm, 1)) + EPS # B x C x C
    frac = num / denom
    frac[frac > (1 - EPS)] = 1 - EPS
    frac[frac < (-1 + EPS)] = -1 + EPS
    output = -torch.mean(torch.acos(frac))
    return output


def cosine_similarity_loss(pred, gold):
    return -F.cosine_similarity(v_pred, v_gold)


def localization_loss(mu, f, a, device=None):
    diff = torch.unsqueeze(mu, 2) - torch.unsqueeze(torch.transpose(f, 1, 2), 1) # B x K x P x C
    dist = torch.sum(diff * diff, 3) # B x K x P
    num = torch.sum(a * dist, 2) # B x K
    denom = torch.sum(a, 2) # B x K
    if device != None:
        eps_tensor = torch.full(denom.size(), EPS).to(device)
    else:
        eps_tensor = torch.full(denom.size(), EPS)
    denom = torch.maximum(denom, eps_tensor)
    div = num / denom
    return torch.sum(div, 1) # B


def repulsion_loss(mu):
    B, K, C = mu.size() 
    unsqueezed_mu = torch.unsqueeze(mu, 2) # B x K x 1 x C
    pairwise_difference = unsqueezed_mu - torch.transpose(unsqueezed_mu, 1, 2) # B x K x K x C
    pairwise_difference = torch.reshape(pairwise_difference, (B, K * K, C)) # B x K * K x C
    dist = torch.sum(pairwise_difference * pairwise_difference, 2) # B x K * K
    output = -torch.mean(dist, 1) # B
    return output

def equilibrium_loss(a):
    return torch.var(torch.mean(a, dim=2), dim = 1) # b

def old_equilibrium_loss(a):
    return torch.mean(torch.var(a, dim=2), dim=1)

def mass_penalization_loss(a):
    b, k, p = a.size()
    inner = 1 - (torch.sum(a, dim=2) / p) # b x k
    return torch.mean(torch.abs(inner), dim=1) # b


def variance_loss(a):
    return -torch.mean(torch.var(a, dim=1), dim=1) # b

def batch_orthonormal_loss(r, device=None):
    ''' Calculate orthonormal_loss. '''
    rtr = torch.matmul(r, r.permute(0, 2, 1)) # B x 3 x 3
    if device != None:
        sub = rtr - torch.eye(3).to(device) # B x 3 x 3
    else:
        sub = rtr - torch.eye(3)
    norm = torch.linalg.norm(sub, dim=(1, 2)) # B
    
    return norm

def regress_loss(mu, f, a, seg_gold, r, gold_r, seg_num_all, args, device, smoothing=True):
    ''' Calculate negative cosine_similarity, localization, and negative repulsion loss. '''
    b, k, p = a.size()
    output = torch.zeros(b).to(device)
    loc_loss = None
    rep_loss = None
    equil_loss = None
    on_loss = None
    cos_loss = None
    if args.localization_weight != 0:
        loc = args.localization_weight * localization_loss(mu, f, a, device)
        output += loc
        loc_loss = torch.mean(loc)
    if args.repulsion_weight != 0:
        rep = args.repulsion_weight * repulsion_loss(mu)
        output += rep
        rep_loss = torch.mean(rep)
    if args.equilibrium_weight != 0:
        equil = args.equilibrium_weight * equilibrium_loss(a)
        output += equil
        equil_loss = torch.mean(equil)
    if args.orthonormal_weight != 0:
        on = args.orthonormal_weight * batch_orthonormal_loss(r, device)
        output += on
        on_loss = torch.mean(on)

    output = torch.mean(output)

    if args.cal_weight != 0:
        seg_pred = a.permute(0, 2, 1).contiguous()
        output = args.cal_weight * cal_loss(seg_pred.view(-1, seg_num_all), seg_gold.view(-1,1).squeeze(), smoothing)
        
    return output, loc_loss, rep_loss, equil_loss, on_loss


def vector_loss(vectors):
    ''' Calculate angle_repulsion_loss and unit norm loss. '''
    ang_rep = angle_repulsion_loss(vectors)
    norm = unit_norm_loss(vectors)
    output = ang_rep + norm
    return output, ang_rep, norm


def orthonormal_loss(r, device=None):
    ''' Calculate orthonormal_loss. '''
    norm = batch_orthonormal_loss(r, device)
    output = torch.mean(norm) # 1
    
    return output


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
