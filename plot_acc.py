"""
python visualize_loss.py /sailhome/sidhikab/vnn-articulated/dgcnn/results/partseg/lr_eqcnn_unet_nonsymmetrized_1024 lr_eqcnn_unet_nonsymmetrized_1024 --type accuracy
"""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    path = "results/partseg/{}/run.log"
    data = {}
    models = ["dgcnn", "vnn", "complex_only", "shell_only", "oavnn"]
    data_types = ["train_epoch", "train_acc", "test_epoch", "test_acc"]
    for model in models:
        data[model] = {}

    dgcnn_train_epoch = []
    dgcnn_train_acc = []
    dgcnn_test_epoch = []
    dgcnn_test_acc = []
    for f in ['dgcnn_128_run0', 'dgcnn_128_run1', 'dgcnn_128_run2']:
        train_epoch = []
        train_acc = []
        test_epoch = []
        test_acc = []
        with open(path.format(f)) as fp:
            for line in fp:
                train_prefix = "Train"
                test_prefix = "Test"
                if line[:len(train_prefix)] == train_prefix or line[:len(test_prefix)] == test_prefix:
                    epoch, loss, acc, avg_acc, iou = line.strip().split(',')
                    epoch_num = int(epoch.split()[-1])
                    acc_num = float(acc.split()[-1])
                    if line[:len(train_prefix)] == train_prefix:
                        train_epoch.append(epoch_num)
                        train_acc.append(acc_num)
                    else:
                        test_epoch.append(epoch_num)
                        test_acc.append(acc_num)

        dgcnn_train_epoch.append(np.expand_dims(np.array(train_epoch), axis=0))
        dgcnn_train_acc.append(np.expand_dims(np.array(train_acc), axis=0))
        dgcnn_test_epoch.append(np.expand_dims(np.array(test_epoch), axis=0))
        dgcnn_test_acc.append(np.expand_dims(np.array(test_acc), axis=0))

    dgcnn_train_epoch = np.concatenate(dgcnn_train_epoch, axis=0).mean(0)
    dgcnn_train_acc = np.concatenate(dgcnn_train_acc, axis=0).mean(0)
    dgcnn_test_epoch = np.concatenate(dgcnn_test_epoch, axis=0).mean(0)
    dgcnn_test_acc = np.concatenate(dgcnn_test_acc, axis=0).mean(0)

    print('DGCNN: train = {}, test = {}'.format(dgcnn_train_acc[-1], dgcnn_test_acc[-1]))


    vnn_train_epoch = []
    vnn_train_acc = []
    vnn_test_epoch = []
    vnn_test_acc = []
    for f in ['vnn_128_run0', 'vnn_128_run1']:
        train_epoch = []
        train_acc = []
        test_epoch = []
        test_acc = []
        with open(path.format(f)) as fp:
            for line in fp:
                train_prefix = "Train"
                test_prefix = "Test"
                if line[:len(train_prefix)] == train_prefix or line[:len(test_prefix)] == test_prefix:
                    epoch, loss, acc, avg_acc, iou = line.strip().split(',')
                    epoch_num = int(epoch.split()[-1])
                    acc_num = float(acc.split()[-1])
                    if line[:len(train_prefix)] == train_prefix:
                        train_epoch.append(epoch_num)
                        train_acc.append(acc_num)
                    else:
                        test_epoch.append(epoch_num)
                        test_acc.append(acc_num)

        vnn_train_epoch.append(np.expand_dims(np.array(train_epoch), axis=0))
        vnn_train_acc.append(np.expand_dims(np.array(train_acc), axis=0))
        vnn_test_epoch.append(np.expand_dims(np.array(test_epoch), axis=0))
        vnn_test_acc.append(np.expand_dims(np.array(test_acc), axis=0))

    vnn_train_epoch = np.concatenate(vnn_train_epoch, axis=0).mean(0)
    vnn_train_acc = np.concatenate(vnn_train_acc, axis=0).mean(0)
    vnn_test_epoch = np.concatenate(vnn_test_epoch, axis=0).mean(0)
    vnn_test_acc = np.concatenate(vnn_test_acc, axis=0).mean(0)

    print('VNN: train = {}, test = {}'.format(vnn_train_acc[-1], vnn_test_acc[-1]))

    oavnn_train_epoch = []
    oavnn_train_acc = []
    oavnn_test_epoch = []
    oavnn_test_acc = []
    for f in ['oavnn_128_run0', 'oavnn_128_run1', 'oavnn_128_run2']:
        train_epoch = []
        train_acc = []
        test_epoch = []
        test_acc = []
        with open(path.format(f)) as fp:
            for line in fp:
                train_prefix = "Train"
                test_prefix = "Test"
                if line[:len(train_prefix)] == train_prefix or line[:len(test_prefix)] == test_prefix:
                    epoch, loss, acc, avg_acc, iou = line.strip().split(',')
                    epoch_num = int(epoch.split()[-1])
                    acc_num = float(acc.split()[-1])
                    if line[:len(train_prefix)] == train_prefix:
                        train_epoch.append(epoch_num)
                        train_acc.append(acc_num)
                    else:
                        test_epoch.append(epoch_num)
                        test_acc.append(acc_num)

        oavnn_train_epoch.append(np.expand_dims(np.array(train_epoch), axis=0))
        oavnn_train_acc.append(np.expand_dims(np.array(train_acc), axis=0))
        oavnn_test_epoch.append(np.expand_dims(np.array(test_epoch), axis=0))
        oavnn_test_acc.append(np.expand_dims(np.array(test_acc), axis=0))

    oavnn_train_epoch = np.concatenate(oavnn_train_epoch, axis=0).mean(0)
    oavnn_train_acc = np.concatenate(oavnn_train_acc, axis=0).mean(0)
    oavnn_test_epoch = np.concatenate(oavnn_test_epoch, axis=0).mean(0)
    oavnn_test_acc = np.concatenate(oavnn_test_acc, axis=0).mean(0)

    print('OAVNN: train = {}, test = {}'.format(oavnn_train_acc[-1], oavnn_test_acc[-1]))

    shell_train_epoch = []
    shell_train_acc = []
    shell_test_epoch = []
    shell_test_acc = []
    for f in ['shell_only_128_run0', 'shell_only_128_run1', 'vnn_shell_channels_128_extended']:
        train_epoch = []
        train_acc = []
        test_epoch = []
        test_acc = []
        with open(path.format(f)) as fp:
            for line in fp:
                train_prefix = "Train"
                test_prefix = "Test"
                if line[:len(train_prefix)] == train_prefix or line[:len(test_prefix)] == test_prefix:
                    epoch, loss, acc, avg_acc, iou = line.strip().split(',')
                    epoch_num = int(epoch.split()[-1])
                    acc_num = float(acc.split()[-1])
                    if line[:len(train_prefix)] == train_prefix:
                        train_epoch.append(epoch_num)
                        train_acc.append(acc_num)
                    else:
                        test_epoch.append(epoch_num)
                        test_acc.append(acc_num)

        shell_train_epoch.append(np.expand_dims(np.array(train_epoch), axis=0))
        shell_train_acc.append(np.expand_dims(np.array(train_acc), axis=0))
        shell_test_epoch.append(np.expand_dims(np.array(test_epoch), axis=0))
        shell_test_acc.append(np.expand_dims(np.array(test_acc), axis=0))

    shell_train_epoch = np.concatenate(shell_train_epoch, axis=0).mean(0)
    shell_train_acc = np.concatenate(shell_train_acc, axis=0).mean(0)
    shell_test_epoch = np.concatenate(shell_test_epoch, axis=0).mean(0)
    shell_test_acc = np.concatenate(shell_test_acc, axis=0).mean(0)

    print('Shell-Only: train = {}, test = {}'.format(shell_train_acc[-1], shell_test_acc[-1]))


    complex_train_epoch = []
    complex_train_acc = []
    complex_test_epoch = []
    complex_test_acc = []
    for f in ['complex_only_128_run0', 'complex_only_128_run1', 'complex_only_128_run2']:
        train_epoch = []
        train_acc = []
        test_epoch = []
        test_acc = []
        with open(path.format(f)) as fp:
            for line in fp:
                train_prefix = "Train"
                test_prefix = "Test"
                if line[:len(train_prefix)] == train_prefix or line[:len(test_prefix)] == test_prefix:
                    epoch, loss, acc, avg_acc, iou = line.strip().split(',')
                    epoch_num = int(epoch.split()[-1])
                    acc_num = float(acc.split()[-1])
                    if line[:len(train_prefix)] == train_prefix:
                        train_epoch.append(epoch_num)
                        train_acc.append(acc_num)
                    else:
                        test_epoch.append(epoch_num)
                        test_acc.append(acc_num)

        complex_train_epoch.append(np.expand_dims(np.array(train_epoch), axis=0))
        complex_train_acc.append(np.expand_dims(np.array(train_acc), axis=0))
        complex_test_epoch.append(np.expand_dims(np.array(test_epoch), axis=0))
        complex_test_acc.append(np.expand_dims(np.array(test_acc), axis=0))

    complex_train_epoch = np.concatenate(complex_train_epoch, axis=0).mean(0)
    complex_train_acc = np.concatenate(complex_train_acc, axis=0).mean(0)
    complex_test_epoch = np.concatenate(complex_test_epoch, axis=0).mean(0)
    complex_test_acc = np.concatenate(complex_test_acc, axis=0).mean(0)

    print('Complex-Only: train = {}, test = {}'.format(complex_train_acc[-1], complex_test_acc[-1]))

    fig, ax = plt.subplots()
    plt.plot(dgcnn_train_epoch, dgcnn_train_acc, label='DGCNN', color='#1f77b4')
    plt.plot(dgcnn_test_epoch, dgcnn_test_acc, '--', color='#1f77b4')
    plt.plot(vnn_train_epoch, vnn_train_acc, label='VNN', color='#ff7f0e')
    plt.plot(vnn_test_epoch, vnn_test_acc, '--', color='#ff7f0e')
    plt.plot(shell_train_epoch, shell_train_acc, label='Shell-Only', color='#d62728')
    plt.plot(shell_test_epoch, shell_test_acc, '--', color='#d62728')
    plt.plot(complex_train_epoch, complex_train_acc, label='Complex-Only', color='#9467bd')
    plt.plot(complex_test_epoch, complex_test_acc, '--', color='#9467bd')
    plt.plot(oavnn_train_epoch, oavnn_train_acc, label='OAVNN', color='#2ca02c')
    plt.plot(oavnn_test_epoch, oavnn_test_acc, '--', color='#2ca02c')
    plt.title('Left-Right Segmentation Results')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    ax.legend()
    fig.savefig('accuracy.png')


if __name__ == "__main__":
    main()