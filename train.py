import os
import argparse
import time
import shutil
import pdb

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from skimage.color import label2rgb
import matplotlib.pyplot as plt

import SimpleITK as sitk
import setproctitle
import numpy as np
from skimage import transform
from sklearn.model_selection import KFold
from scipy.spatial.distance import directed_hausdorff

from model import AMRSegNet_noalpha
from utils.logger import Logger
from utils.loss import DiceLoss, SurfaceLoss, TILoss
from dataset.lung_dataset import Lung_dataset, RandomHorizontalFlip, Crop, RandomRotation, ToTensor, Normalize

plt.switch_backend('agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def confusion(y_pred, y_true):
    '''
    get precision and recall
    '''
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    y_pred = y_pred.float()
    y_true = y_true.float()
    smooth = 1.
    # y_pred_pos = np.clip(y_pred, 0, 1)
    y_pred_pos = y_pred
    y_pred_neg = 1 - y_pred_pos
    # y_pos = np.clip(y_true, 0, 1)
    y_pos = y_true
    y_neg = 1 - y_true

    tp = torch.dot(y_pos, y_pred_pos)
    fp = torch.dot(y_neg, y_pred_pos)
    fn = torch.dot(y_pos, y_pred_neg)

    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return prec, recall

def compute_hausdorff(y_pred, y_true):
    '''
    get directed hausdorff
    '''
    # pdb.set_trace()
    shape = y_pred.shape
    y_true = np.reshape(y_true, shape)
    hausdorff_distance = max(directed_hausdorff(y_pred, y_true)[0], directed_hausdorff(y_true, y_pred)[0])

    return hausdorff_distance

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'adam':
        if epoch < 40:
            lr = 1e-3
        elif epoch == 60:
            lr = 1e-4
        elif epoch == 70:
            lr = 1e-5
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


def datestr():
    now = time.localtime()
    return '{}_{:02}_{:02}_{:02}:{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=6)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=70)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-i', '--inference', default='', type=str, metavar='PATH',
                        help='run inference on data set and save results')
    parser.add_argument('-di', '--dwiinference', default='', type=str, metavar='PATH',
                        help='run inference on data set and save results')
    parser.add_argument('-t', '--target', default='', type=str, metavar='PATH',
                        help='target to get dice on data set and save results')
    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float, metavar='W',
                        help='weight decay (default: 1e-8)')
    parser.add_argument('--save')
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))
    # parser.add_argument('--log', default=None, type=str,
    #                    help='log dir to save information for tensorboard')
    # parser.add_argument('--comment', type=str, default=None,
    #                    help='comment for tensorboardX')

    args = parser.parse_args()
    return args


def train(args, epoch, model, train_loader, optimizer, trainF, loss_fn, writer):
    model.train()
    nProcessed = 0
    nTrain = len(train_loader.dataset)
    loss_list = []
    print('--------------------Epoch{}------------------------'.format(epoch))
    for batch_idx, sample in enumerate(train_loader):
        # read data
        data, data2, target = sample['image'], sample['image_b'], sample['target']
        # pdb.set_trace()
        if args.cuda:
            data, data2, target = data.cuda(), data2.cuda(), target.cuda()
        data, data2, target = Variable(data), Variable(data2), Variable(target, requires_grad=False)

        # print('data.shape: ', data.shape)
        # print('data2.shape: ', data2.shape)
        # feed to model
        output = model(data, data2)
        target = target.view(output.shape[0], target.numel() // output.shape[0])

        # loss
        loss = loss_fn['dice_loss'](output, target)

        target = target.long()

        dice, jaccard = DiceLoss.dice_coeficient(output > 0.5, target)
        precision, recall = confusion(output > 0.5, target)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show some result on tensorboard
        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / len(train_loader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
            loss.item()))
        print('Jaccard index: %6f, soft dice: %6f' % (jaccard, dice))

        # writer.add_scalar('train_loss/epoch', loss, partialEpoch)
        trainF.write('{},{},{}\n'.format(partialEpoch, loss.item(), loss.item()))
        trainF.flush()
        # show images on tensorboard
        with torch.no_grad():
            shape = [data.shape[0], 1, data.shape[2], data.shape[3]]
            if batch_idx % 4 == 0:
                # img = make_grid(data, padding=20).cpu().numpy().transpose(1, 2, 0)[0]
                # pdb.set_trace()
                data = (data * 0.5 + 0.5)
                data2 = (data2 * 0.5 + 0.5)

                img = make_grid(data, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                img2 = make_grid(data2, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                # print('img.shape', img.shape)
                target = target.view(shape)
                target = target.float()
                gt = make_grid(target, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                # _, pre = output_softmax.max(1)

                pre = output > 0.5
                pre = pre.float()
                # pdb.set_trace()
                pre = pre.view(shape)
                pre = make_grid(pre, padding=20).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
                # pdb.set_trace()
                gt_img = label2rgb(gt, img, bg_label=0)
                pre_img = label2rgb(pre, img, bg_label=0)
                gt_img2 = label2rgb(gt, img2, bg_label=0)
                # pdb.set_trace()

                fig = plt.figure()
                ax = fig.add_subplot(311)
                ax.imshow(gt_img)
                ax.set_title('T2 ground truth')
                ax = fig.add_subplot(312)
                ax.imshow(pre_img)
                ax.set_title('prediction')
                ax = fig.add_subplot(313)
                ax.imshow(gt_img2)
                ax.set_title('DWI ground truth')
                fig.tight_layout()

                writer.add_figure('train result', fig, epoch)
                fig.clear()
    loss_list.append(loss.item())

    return np.mean(loss_list)


def test(args, epoch, model, test_loader, optimizer, testF, loss_fn, logger, writer):
    model.eval()
    mean_dice = []
    mean_jaccard = []
    mean_precision = []
    mean_recall = []
    mean_hausdorff = []

    with torch.no_grad():
        for sample in test_loader:
            data, data2, target = sample['image'], sample['image_b'], sample['target']
            if args.cuda:
                data, data2, target = data.cuda(), data2.cuda(), target.cuda()
            data, data2, target = Variable(data), Variable(data2), Variable(target, requires_grad=False)

            output = model(data, data2)

            # target = target.view(target.numel())
            # loss = loss_fn['dice_loss'](output, target[:,:,7:-7,7:-7])
            # dice = 1 - loss
            # m = nn.Softmax(dim=1)
            # output = m(output)

            # pdb.set_trace()
            # Hausdorff Distance
            hausdorff_distance = compute_hausdorff(output.cpu().numpy(), target.cpu().numpy())
            # Dice coefficient
            dice, jaccard = DiceLoss.dice_coeficient(output, target)
            precision, recall = confusion(output, target)

            mean_precision.append(precision.item())
            mean_recall.append(recall.item())
            mean_dice.append(dice.item())
            mean_jaccard.append(jaccard.item())
            mean_hausdorff.append(hausdorff_distance)

        # show the last sample
        shape = [data.shape[0], 1, data.shape[2], data.shape[3]]
        if epoch % 1 == 0:
            # img = make_grid(data, padding=20).cpu().numpy().transpose(1, 2, 0)[0]
            data = (data * 0.5 + 0.5)
            data2 = (data2 * 0.5 + 0.5)
            img = make_grid(data, padding=20).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
            img2 = make_grid(data2, padding=20).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
            # print('img.shape', img.shape)
            target = target.view(shape)
            target = target.float()
            gt = make_grid(target, padding=20).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
            # _, pre = output_softmax.max(1)
            pre = output > 0.5
            pre = pre.float()
            pre = pre.view(shape)
            pre = make_grid(pre, padding=20).cpu().numpy().transpose(1, 2, 0)[:, :, 0]

            gt_img = label2rgb(gt, img, bg_label=0)
            pre_img = label2rgb(pre, img, bg_label=0)
            gt_img2 = label2rgb(gt, img2, bg_label=0)

            fig = plt.figure()
            ax = fig.add_subplot(311)
            ax.imshow(gt_img)
            ax.set_title('T2 ground truth')
            ax = fig.add_subplot(312)
            ax.imshow(pre_img)
            ax.set_title('prediction')
            ax = fig.add_subplot(313)
            ax.imshow(gt_img2)
            ax.set_title('DWI ground truth')
            fig.tight_layout()

            writer.add_figure('test result', fig, epoch)
            fig.clear()

        writer.add_scalar('fold4_test_dice/epoch', np.mean(mean_dice), epoch)
        writer.add_scalar('fold4_test_jaccard/epoch', np.mean(mean_jaccard), epoch)
        writer.add_scalar('fold4_test_precisin/epoch', np.mean(mean_precision), epoch)
        writer.add_scalar('fold4_test_recall/epoch', np.mean(mean_recall), epoch)
        writer.add_scalar('fold4_hausdorff_distance/epoch', np.mean(mean_hausdorff), epoch)

        print('test mean_dice: ', np.mean(mean_dice))
        print('test mean jaccard: ', np.mean(mean_jaccard))
        print('mean_dice_length ', len(mean_dice))
        testF.write('{},{},{},{}\n'.format(epoch, np.mean(mean_dice), np.mean(mean_precision), np.mean(mean_recall)))
        testF.flush()
        return np.mean(mean_dice), np.mean(mean_recall), np.mean(mean_precision)


def inference(args, loader, model):
    src = args.inference
    model.eval()
    dice_list = []
    mean_precision = []
    mean_recall = []
    mean_hausdorff = []
    mean_jaccard = []

    with torch.no_grad():
        for num, sample in enumerate(loader):
            data, data2, target = sample['image'], sample['image_b'], sample['target']
            if args.cuda:
                data, data2, target = data.cuda(), data2.cuda(), target.cuda()
            data, data2, target = Variable(data), Variable(data2), Variable(target)

            output = model(data, data2)

            loss, jaccard = DiceLoss.dice_coeficient(output, target)
            precision, recall = confusion(output, target)
            hausdorff_distance = compute_hausdorff(output.cpu().numpy(), target.cpu().numpy())
            # dice = loss.cpu().numpy().astype(np.float32)
            dice = loss.cpu().numpy()
            dice_list.append(dice)
            mean_precision.append(precision.item())
            mean_recall.append(recall.item())
            mean_hausdorff.append(hausdorff_distance)
            mean_jaccard.append(jaccard.item())

            data = (data * 0.5 + 0.5)
            data2 = (data2 * 0.5 + 0.5)
            img = make_grid(data, padding=20).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
            img2 = make_grid(data2, padding=20).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
            target = target.view(data.shape)
            target = target.float()
            gt = make_grid(target, padding=20).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
            # _, pre = output_softmax.max(1)
            pre = output > 0.5
            pre = pre.float()
            pre = pre.view(data.shape)
            pre = make_grid(pre, padding=20).cpu().numpy().transpose(1, 2, 0)[:, :, 0]

            gt_img = label2rgb(gt, img, bg_label=0)
            pre_img = label2rgb(pre, img, bg_label=0)
            gt_img2 = label2rgb(gt, img2, bg_label=0)

            fig = plt.figure()
            ax = fig.add_subplot(231)
            ax.imshow(gt_img)
            ax.set_title('T2 ground truth')
            ax.axis('off')
            ax = fig.add_subplot(233)
            ax.imshow(pre_img)
            ax.set_title('prediction')
            ax.axis('off')
            ax = fig.add_subplot(232)
            ax.imshow(gt_img2)
            ax.set_title('DWI ground truth')
            ax.axis('off')
            ax = fig.add_subplot(234)
            ax.imshow(img)
            ax.set_title('T2 image')
            ax.axis('off')
            ax = fig.add_subplot(235)
            ax.imshow(img2)
            ax.set_title('DWI image')
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(
                '/home/Multi_Modality/data/fold5/inference/AMRSegNet_noalpha/%d_%4f.png' % (
                num, dice))

            print('processing {}/{}\r dice:{}'.format(num, len(loader.dataset), dice))

        mean_jaccard = np.array(mean_jaccard).mean()
        mean_dice = np.array(dice_list).mean()
        std_dice = np.std(np.array(dice_list))
        mean_recall = np.mean(mean_recall)
        mean_precision = np.mean(mean_precision)
        mean_hausdorff = np.mean(mean_hausdorff)
        F1_score = 2 * mean_recall * mean_precision / (mean_recall + mean_precision)

        print('mean_jaccard: %4f' % mean_jaccard)
        print('mean_dice: %4f' % mean_dice)
        print('std: %4f' % std_dice)
        print('mean_recall: %4f' % mean_recall)
        print('mean_precision: %4f' % mean_precision)
        print('F1_score: ', F1_score)
        print('mean_hausdorff: %4f' % mean_hausdorff)



def make_one_hot(label_tensor, num_classes):
    shape = np.array(label_tensor.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    output = torch.zeros(shape)
    output = output.scatter_(1, label_tensor.cpu(), 1)

    return output


def main():
    #############
    # init args #
    #############
    train_T2_path     = '/home/Multi_Modality/data/fold4/train/T2_aug'
    train_target_path = '/home/Multi_Modality/data/fold4/train/label_aug'
    train_DWI_path    = '/home/Multi_Modality/data/fold4/train/DWI_aug'

    test_T2_path      = '/home/Multi_Modality/data/fold4/test/test_T2'
    test_target_path  = '/home/Multi_Modality/data/fold4/test/test_label'
    test_DWI_path     = '/home/Multi_Modality/data/fold4/test/test_DWI'

    args = get_args()

    best_prec1 = 0.
    best_prec2 = 0.
    best_prec3 = 0.

    args.cuda = torch.cuda.is_available()
    if args.inference == '':
        args.save = args.save or 'work/AMRSegNet_fold4_SE.{}'.format(datestr())
    else:
        args.save = args.save or 'work/AMRSegNet_fold4_SE_inference.{}'.format(datestr())

    weight_decay = args.weight_decay
    setproctitle.setproctitle(args.save)

    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)

    if args.inference == '':
    # writer for tensorboard
        if args.save and args.inference == '':
            idx = args.save.rfind('/')
            log_dir = 'runs' + args.save[idx:]
            print('log_dir', log_dir)
            writer = SummaryWriter(log_dir)
        else:
            writer = SummaryWriter()
    else:
        idx = args.save.rfind('/')
        log_dir = 'runs' + args.save[idx:]
        print('log_dir', log_dir)
        writer = SummaryWriter(log_dir)

    #########################
    # building  AMRSegNet   #
    #########################
    print("building AMRSegNet-----")
    batch_size = args.ngpu * args.batchSz
    # model = unet.UNet(relu=False)
    model = AMRSegNet_noalpha.AMRSegNet()

    x = torch.zeros((1, 1, 256, 256))
    writer.add_graph(model, (x, x))

    if args.cuda:
        model = model.cuda()

    model = nn.parallel.DataParallel(model, list(range(args.ngpu)))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)

    print('Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # if args.cuda:
    #     model = model.cuda()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # define a logger and write information
    logger = Logger(os.path.join(args.save, 'log.txt'))
    logger.print3('batch size is %d' % args.batchSz)
    logger.print3('nums of gpu is %d' % args.ngpu)
    logger.print3('num of epochs is %d' % args.nEpochs)
    logger.print3('start-epoch is %d' % args.start_epoch)
    logger.print3('weight-decay is %e' % args.weight_decay)
    logger.print3('optimizer is %s' % args.opt)

    ################
    # prepare data #
    ################
    # train_transform = transforms.Compose([RandomHorizontalFlip(p=0.7),
    #                                       RandomRotation(30),
    #                                       Crop(),
    #                                       ToTensor(),
    #                                       Normalize(0.5, 0.5)])
    train_transform = transforms.Compose([Crop(), ToTensor(), Normalize(0.5, 0.5)])
    # train_transform = transforms.Compose([Crop(), ToTensor(), Normalize(0.5, 0.5)])
    test_transform = transforms.Compose([Crop(), ToTensor(), Normalize(0.5, 0.5)])

    # inference dataset
    if args.inference != '':
        if not args.resume:
            print("args.resume must be set to do inference")
            exit(1)
        kwargs = {'num_workers': 0} if args.cuda else {}
        T2_src = args.inference
        DWI_src = args.dwiinference
        tar = args.target

        inference_batch_size = 1
        dataset = Lung_dataset(image_path=T2_src, image2_path=DWI_src, target_path=tar, transform=test_transform)
        loader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, **kwargs)
        inference(args, loader, model)

        return

    # tarin dataset
    print("loading train set --- ")
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    train_set = Lung_dataset(image_path=train_T2_path,
                             image2_path=train_DWI_path,
                             target_path=train_target_path,
                             transform=train_transform)
    test_set = Lung_dataset(image_path=test_T2_path,
                            image2_path=test_DWI_path,
                            target_path=test_target_path,
                            transform=test_transform, mode='test')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, **kwargs)

    # class_weights
    target_mean = train_set.get_target_mean()
    bg_weight = target_mean / (1. + target_mean)
    fg_weight = 1. - bg_weight
    class_weights = torch.FloatTensor([bg_weight, fg_weight])
    if args.cuda:
        class_weights = class_weights.cuda()

    #############
    # optimizer #
    #############
    lr = 0.7*1e-2
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-1,
                              momentum=0.99, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)

    # loss function
    loss_fn = {}
    loss_fn['surface_loss'] = SurfaceLoss()
    loss_fn['ti_loss'] = TILoss()
    loss_fn['dice_loss'] = DiceLoss()
    loss_fn['l1_loss'] = nn.L1Loss()
    loss_fn['CELoss'] = nn.CrossEntropyLoss()

    ############
    # training #
    ############
    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    err_best = 0.

    for epoch in range(1, args.nEpochs + 1):
        # adjust_opt(args.opt, optimizer, epoch)
        if epoch > 20:
            lr = 1e-3
        if epoch > 30:
            lr = 1e-4
        if epoch > 50:
            lr = 1e-5
        # if epoch > 40:
        #     lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        mean_loss = train(args, epoch, model, train_loader, optimizer, trainF, loss_fn, writer)
        dice, recall, precision = test(args, epoch, model, test_loader, optimizer, testF, loss_fn, logger, writer)
        writer.add_scalar('fold4_train_loss/epoch', mean_loss, epoch)

        is_best1, is_best2, is_best3 = False, False, False
        if dice > best_prec1:
            is_best1 = True
            best_prec1 = dice
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1},
                        is_best1, args.save, "AMRSegNet_dice")

    trainF.close()
    testF.close()

    writer.close()


if __name__ == '__main__':
    main()
