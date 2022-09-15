#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: main_partseg.py
@Time: 2019/12/31 11:17 AM
sbatch 3gpu.sbatch /orion/u/sidhikab/miniconda/envs/3d/bin/python run_models.py --exp_name dgcnn_128_run0 --model dgcnn --rot so3 --class_choice airplane --dataset shapenet_single_class --num_points 128
sbatch 3gpu.sbatch /orion/u/sidhikab/miniconda/envs/3d/bin/python run_models.py --exp_name vnn_128_run0 --model vnn --rot so3 --class_choice airplane --dataset shapenet_single_class --num_points 128
sbatch 3gpu.sbatch /orion/u/sidhikab/miniconda/envs/3d/bin/python run_models.py --exp_name complex_only_128_run0 --model complex_only --rot so3 --class_choice airplane --dataset shapenet_single_class --num_points 128
sbatch 3gpu.sbatch /orion/u/sidhikab/miniconda/envs/3d/bin/python run_models.py --exp_name shell_only_128_run0 --model shell_only --rot so3 --class_choice airplane --dataset shapenet_single_class --num_points 128
sbatch 3gpu.sbatch /orion/u/sidhikab/miniconda/envs/3d/bin/python run_models.py --exp_name oavnn_128_run0 --model oavnn --rot so3 --class_choice airplane --dataset shapenet_single_class --num_points 128
python main_final.py --exp_name test --model oavnn --rot so3 --class_choice airplane --dataset shapenet_single_class --num_points 128
python main_combined.py --exp_name test --model eqcnn --rot so3 --class_choice airplane --dataset shapenetfourfb --num_points 512
export HDF5_USE_FILE_LOCKING=FALSE
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import *
from model import *
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import pickle

from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations


def _init_():
    if not os.path.exists('results/partseg'):
        os.makedirs('results/partseg')
    if not os.path.exists('results/partseg/' + args.exp_name):
        os.makedirs('results/partseg/' + args.exp_name)
    if not os.path.exists('results/partseg/' + args.exp_name + '/' + 'models'):
        os.makedirs('results/partseg/' + args.exp_name + '/' + 'models')
    os.system('cp main_partseg.py results/partseg' + '/' + args.exp_name + '/' + 'main_partseg.py.backup')
    os.system('cp model_equi.py results/partseg' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py results/partseg' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py results/partseg' + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_shape_IoU(pred_np, seg_np, label, class_choice, args):
    if args.dataset == 'shapenetpart' or args.dataset == "body" or args.dataset == "wing" or args.dataset == 'shapenet_single_class':
        seg_num = [2, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    elif args.dataset == 'shapenetfourfb' or args.dataset == 'shapenetfourtb' or args.dataset == "shapenetfourfbnonsymmetrized" or args.dataset == "bodywing" or args.dataset == "shapenetcustomsep":
        seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    elif args.dataset == 'shapenetsym' or args.dataset == 'shapeneteight' or args.dataset == "shapenetpartsym":
        seg_num = [8, 4, 4, 8, 8, 6, 6, 4, 8, 4, 12, 4, 6, 6, 6, 6]
    elif args.dataset == 'shapenettwelve':
        seg_num = [12, 4, 4, 8, 8, 6, 6, 4, 8, 4, 12, 4, 6, 6, 6, 6]
    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def get_lr_seg(data):
    # make symmetric
    B, C, _ = data.size()
    seg = torch.zeros((B, C))
    seg[data[:, :, 2] > 0] = 1
    seg[data[:, :, 2] <= 0] = 0
    return seg


def train(args, io):
    if args.dataset == 'shapenet_single_class':
        train_dataset = ShapeNetSingleClass(partition='trainval', num_points=args.num_points,
                                            flatten_dim=args.flatten_dim, class_choice=args.class_choice)
        if (len(train_dataset) < 100):
            drop_last = False
        else:
            drop_last = True
        train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True,
                                  drop_last=drop_last)
        test_loader = DataLoader(
            ShapeNetSingleClass(partition='test', num_points=args.num_points, flatten_dim=args.flatten_dim,
                                class_choice=args.class_choice),
            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        seg_num_all = 2
    elif args.dataset == 'shapenetpart':
        train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points, class_choice=args.class_choice)
        if (len(train_dataset) < 100):
            drop_last = False
        else:
            drop_last = True
        train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True,
                                  drop_last=drop_last)
        test_loader = DataLoader(
            ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice),
            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        seg_num_all = 2
    else:
        train_dataset = ShapeNetCustom(args.data_path, partition='trainval', num_points=args.num_points,
                                       class_choice=args.class_choice)
        if (len(train_dataset) < 100):
            drop_last = False
        else:
            drop_last = True
        train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True,
                                  drop_last=drop_last)
        test_loader = DataLoader(ShapeNetCustom(args.data_path, partition='test', num_points=args.num_points,
                                                class_choice=args.class_choice),
                                 num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        seg_num_all = 2

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    seg_start_index = train_loader.dataset.seg_start_index
    if args.model == "dgcnn":  # dgcnn_128, num_points=64
        model = DGCNN(args, seg_num_all).to(device)
    elif args.model == "vnn":  # vnn_128, num_points=64
        model = VNN(args, seg_num_all).to(device)
    elif args.model == "complex_only":  # complex_only_128, num_points=128
        model = Complex_Only(args, seg_num_all).to(device)
    elif args.model == "shell_only":  # shell_only_128, num_points=128
        model = Shell_Only(args, seg_num_all).to(device)
    elif args.model == "oavnn":  # oavnn_128, num_points=128
        model = OAVNN(args, seg_num_all).to(device)

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epochs):
        print("start first epoch")
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, label, seg in train_loader:
            seg = get_lr_seg(data)
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(data.shape[0]) * 360, axis="Z", degrees=True, device=device)
            elif args.rot == 'so3':
                R = random_rotations(data.shape[0])
                unrot_R = torch.inverse(R)
                trot = Rotate(R=R, device=device)
                tunrot = Rotate(R=unrot_R, device=device)

            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            if trot is not None:
                data = trot.transform_points(data)

            seg = seg.type(torch.int64)

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(label.reshape(-1))
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice, args)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch,
                                                                                                  train_loss * 1.0 / count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        for data, label, seg in test_loader:
            seg = get_lr_seg(data)
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(data.shape[0]) * 360, axis="Z", degrees=True, device=device)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(data.shape[0]), device=device)

            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            if trot is not None:
                data = trot.transform_points(data)

            seg = seg.type(torch.int64)

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label_seg.append(label.reshape(-1))
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_label_seg = np.concatenate(test_label_seg)
        test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice, args)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss * 1.0 / count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'results/partseg/%s/models/model.t7' % args.exp_name)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='dgcnn_vnn', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='eqcnn', metavar='N',
                        help='Model to use')
    parser.add_argument('--dataset', type=str, default='', metavar='N')
    parser.add_argument('--data_path', type=str, default='', metavar='N')
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--num_shells', type=int, default=4,
                        help='num of shells to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--rot', type=str, default='aligned', metavar='N',
                        choices=['aligned', 'z', 'so3'],
                        help='Rotation augmentation to input data')
    parser.add_argument('--use_batchnorm', type=str, default='norm', metavar='N',
                        choices=['norm', 'norm_log', 'none'],
                        help='VNN only: use batch normalization on given invariant feature.')
    parser.add_argument('--pooling', type=str, default='mean', metavar='N',
                        choices=['mean', 'max'],
                        help='VNN only: pooling method.')
    parser.add_argument('--inv_use_mean', type=bool, default=True,
                        help='VNN only: input global mean to invariant layer.')
    parser.add_argument('--inv_mlp_layers', type=int, default=3, metavar='N',
                        choices=[1, 3],
                        help='VNN only: number of VN layers used in the invariant layer.')
    parser.add_argument('--combine_lin_nonlin', type=bool, default=True,
                        help='VNN only: combine linear layer into nonlinearity.')
    parser.add_argument('--ref_var', dest='reflection_variant', action='store_true')
    parser.add_argument('--ref_invar', dest='reflection_variant', action='store_false')
    parser.set_defaults(reflection_variant=True)
    parser.add_argument('--flatten_dim', type=int, default=-1, metavar='N',
                        choices=[-1, 0, 1, 2],
                        help='VNN only: dimension to flatten.')
    parser.add_argument('--segmentation_axis', type=int, default=2, metavar='N',
                        choices=[0, 1, 2],
                        help='axis over which segmentation is evaluated')
    parser.add_argument('--only_b', dest='reflection_variant', action='store_true')
    parser.add_argument('--a_and_b', dest='reflection_variant', action='store_false')
    parser.set_defaults(b_only=False)
    parser.add_argument('--embedding', dest='reflection_variant', action='store_true')
    parser.add_argument('--no_embedding', dest='reflection_variant', action='store_false')
    parser.set_defaults(use_embedding=False)
    parser.add_argument('--use_x_coord', dest='use_x_coord', action='store_true')
    parser.add_argument('--no_use_x_coord', dest='use_x_coord', action='store_false')
    parser.set_defaults(use_x_coord=True)
    args = parser.parse_args()

    _init_()

    io = IOStream('results/partseg/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    train(args, io)