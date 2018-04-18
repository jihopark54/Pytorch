import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.autograd import Variable

from datasets.load_svhn import Data
from DRAM.DRAMnet import DRAMnet

resume = "./parameter/model_dram.pth.tar"
start_epoch = 0
global_lr = 0.001
evaluate_chk = False
cuda_chk = True
iteration = 40000
best_prec1 = 0
print_freq = 10

def main():

    # define loss function (criterion) and optimizer
    global start_epoch, best_prec1

    model = DRAMnet()

    if cuda_chk:
        model.glimpse_image_conv.cuda()
        model.glimpse_image_fc.cuda()
        model.glimpse_loc.cuda()
        model.recurrent_1.cuda()
        model.recurrent_2.cuda()
        model.emission.cuda()
        model.context.cuda()
        model.classification.cuda()

    trainset = Data('train', 64)
    testset = Data('test', 64)

    criterion = nn.CrossEntropyLoss()
    if cuda_chk:
        criterion.cuda()

    params_dict = dict(model.named_parameters())
    params = []
    for idx, (key, value) in enumerate(params_dict.items()):
        params += [{'params': [value], 'lr': global_lr}]

    optimizer = torch.optim.SGD(params, weight_decay=0.0005)

    # optionally resume from a checkpoint
    if resume is not None:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))


    if evaluate_chk:
        validate(testset, model, criterion)
        return

    for epoch in range(start_epoch, start_epoch + iteration):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(trainset, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(testset, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                         'best_prec1': best_prec1, 'optimizer': optimizer.state_dict()}, is_best, resume)

        print("epoch {} complete".format(epoch))

        trainset.shuffle()
        testset.shuffle()


def train(dataset, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    batch_size = 128
    end = time.time()
    step = 0
    while not dataset.end:
        # measure data loading time
        datax, _, datay = dataset.next(batch_size)

        batch_x = torch.from_numpy(datax)
        batch_y = torch.from_numpy(datay).type(torch.LongTensor)
        if cuda_chk:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y)

        data_time.update(time.time() - end)

        # compute output
        output = model(batch_x)

        # compute loss
        loss = criterion(output[:, 0, :], batch_y[:, 0])

        # measure accuracy and record loss
        prec1 = accuracy(output.data, batch_y.data, topk=(1,)).float()
        losses.update(loss.data[0], batch_x.data.size(0))
        top1.update(prec1.cpu().numpy()[0], batch_x.data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        step += 1
        if step % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, step * batch_size, dataset.length, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def validate(dataset, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    batch_size = 100
    end = time.time()
    step = 0

    while not dataset.end:
        datax, _, datay = dataset.next(batch_size)
        batch_x = torch.from_numpy(datax)
        batch_y = torch.from_numpy(datay).type(torch.LongTensor)
        if cuda_chk:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y)

        # compute output
        output = model(batch_x)
        loss1 = criterion(output[:, 0, :], batch_y[:, 0])
        loss2 = criterion(output[:, 1, :], batch_y[:, 1])
        loss3 = criterion(output[:, 2, :], batch_y[:, 2])
        loss4 = criterion(output[:, 3, :], batch_y[:, 3])
        loss5 = criterion(output[:, 4, :], batch_y[:, 4])

        loss = loss1 + loss2 + loss3 + loss4 + loss5
        # measure accuracy and record loss
        prec1 = accuracy(output.data, batch_y.data, topk=(1,))
        losses.update(loss.data[0], batch_x.data.size(0))
        top1.update(prec1.cpu().numpy()[0], batch_x.data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        step += 1
        if step % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   step * batch_size, dataset.length, batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print('save')
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = global_lr * (0.1 ** (epoch // 80000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 2, True, True)
    pred = pred.view(batch_size, -1)
    correct = pred.eq(target)

    res = correct.view(-1).float().sum(0) / batch_size
    return res


if __name__ == '__main__':
    main()