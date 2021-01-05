#!/usr/bin/env python3
import argparse
import os
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nBatches', type=int)
    parser.add_argument('expDir', type=str)
    args = parser.parse_args()

    trainP = os.path.join(args.expDir, 'train.csv')
    trainData = np.loadtxt(trainP, delimiter=',').reshape(-1, 3)
    testP = os.path.join(args.expDir, 'test.csv')
    testData = np.loadtxt(testP, delimiter=',').reshape(-1, 3)

    N = args.nBatches
    trainI, trainLoss, trainErr = np.split(trainData, [1,2], axis=1)
    trainI, trainLoss, trainErr = [x.ravel() for x in
                                   (trainI, trainLoss, trainErr)]
    trainI_, trainLoss_, trainErr_ = rolling(N, trainI, trainLoss, trainErr)

    testI, testLoss, testErr = np.split(testData, [1,2], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # plt.plot(trainI, trainLoss, label='Train')
    plt.plot(trainI_, trainLoss_, label='Train')
    plt.plot(testI, testLoss, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('ce_dice_loss')
    plt.legend()
    ax.set_yscale('log')
    loss_fname = os.path.join(args.expDir, 'loss.png')
    fig.tight_layout()
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # plt.plot(trainI, trainErr, label='Train')
    plt.plot(trainI_, trainErr_, label='Train')
    plt.plot(testI, testErr, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('dice')
    ax.set_yscale('log')
    plt.legend()
    err_fname = os.path.join(args.expDir, 'dice.png')
    fig.tight_layout()
    plt.savefig(err_fname)
    print('Created {}'.format(err_fname))

    loss_err_fname = os.path.join(args.expDir, 'loss-dice.png')
    os.system('convert +append {} {} {}'.format(loss_fname, err_fname, loss_err_fname))
    print('Created {}'.format(loss_err_fname))

def rolling(N, i, loss, err):
    i_ = i[N-1:]
    K = np.full(N, 1./N)
    loss_ = np.convolve(loss, K, 'valid')
    err_ = np.convolve(err, K, 'valid')
    return i_, loss_, err_

if __name__ == '__main__':
    main()

