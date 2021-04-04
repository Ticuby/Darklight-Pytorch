#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed April 1 13:39:00 2021
This repository is based on the repository at https://github.com/artest08/LateTemporalModeling3DCNN. We thank the authors for the repository.
This repository is authored by Jiajun Chen
We thank the authors for the repository.
"""

import os
import time
import argparse
import shutil
import numpy as np


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

from torch.optim import lr_scheduler
import video_transforms
import models
import datasets
#import swats
from opt.AdamW import AdamW

import csv

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to datset setting files')
parser.add_argument('--dataset', '-d', default='ARID',
                    choices=["ucf101", "hmdb51", "smtV2", "window", "ARID"],
                    help='dataset: ucf101 | hmdb51 | smtV2')

parser.add_argument('--arch', '-a', default='dark_light',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        '(default: dark_light)')

parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=3, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--iter-size', default=16, type=int,
                    metavar='I', help='iter size to reduce memory usage (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)')
parser.add_argument('--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 400)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 1)')
parser.add_argument('--num-seg', default=1, type=int,
                    metavar='N', help='Number of segments in dataloader (default: 1)')
#parser.add_argument('--resume', default='./dene4', type=str, metavar='PATH',
#                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-c', '--continue', dest='contine', action='store_true',
                    help='continue training')
parser.add_argument('-g','--gamma',default=1,type=float,
                    help="the value of gamma")
parser.add_argument('--both-flow',default='True',
                    help='give dark and light flow both')
parser.add_argument('--no-ttention',default=True,action='store_false',help="use attention to instead of linear")

best_prec1 = 0
best_loss = 30
warmUpEpoch = 5


def main():
    global args, best_prec1, model, writer, best_loss, length, width, height, input_size, scheduler, suffix
    args = parser.parse_args()
    training_continue = args.contine
    if not args.no_attention:
        args.arch='dark_light_noAttention'


    suffix = 'ga=%s_b=%s_both_flow=%s' % (args.gamma , args.batch_size , args.both_flow)
    headers = ['epoch', 'top1', 'top5', 'loss']
    with open('train_record_%s.csv' % suffix, 'w', newline='') as f:
        record = csv.writer(f)
        record.writerow(headers)

    with open('validate_record_%s.csv' % suffix, 'w', newline='') as f:
        record = csv.writer(f)
        record.writerow(headers)

    print('work in both_flow = %s, gamma = %s, batch_size = %s'%(args.both_flow, args.gamma, args.batch_size))
    
    input_size = 112
    width = 170
    height = 128

    saveLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    writer = SummaryWriter(saveLocation)
   
    # create model

    if args.evaluate:
        print("Building validation model ... ")
        model = build_model_validate()
        optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    elif training_continue:
        model, startEpoch, optimizer, best_prec1 = build_model_continue()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Continuing with best precision: %.3f and start epoch %d and lr: %f" %(best_prec1,startEpoch,lr))
    else:
        print("Building model with ADAMW... ")
        model = build_model()
        optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
        startEpoch = 0

    
    print("Model %s is loaded. " % (args.arch))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, verbose=True)

    print("Saving everything to directory %s." % (saveLocation))
    dataset='./datasets/ARID_frames'

    cudnn.benchmark = True
    length=64
    # Data transforming
    is_color = True
    scale_ratios = [1.0, 0.875, 0.75, 0.66]
    clip_mean = [0.485, 0.456, 0.406] * args.num_seg * length
    clip_std = [0.229, 0.224, 0.225] * args.num_seg * length

    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)

    train_transform = video_transforms.Compose([
            video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
            video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor(),
            normalize,
        ])

    val_transform = video_transforms.Compose([
            video_transforms.CenterCrop((input_size)),
            video_transforms.ToTensor(),
            normalize,
        ])

    # data loading
    train_setting_file = "train_rgb_split%d.txt" % (args.split)
    train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)
    val_setting_file = "val_rgb_split%d.txt" % (args.split)
    val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))
    #ARID.py
    train_dataset = datasets.__dict__[args.dataset](root=dataset,
                                                    modality="rgb",
                                                    source=train_split_file,
                                                    phase="train",
                                                    is_color=is_color,
                                                    new_length=length,
                                                    new_width=width,
                                                    new_height=height,
                                                    video_transform=train_transform,
                                                    num_segments=args.num_seg,
                                                    gamma=args.gamma)
    
    val_dataset = datasets.__dict__[args.dataset](root=dataset,
                                                  modality="rgb",
                                                  source=val_split_file,
                                                  phase="val",
                                                  is_color=is_color,
                                                  new_length=length,
                                                  new_width=width,
                                                  new_height=height,
                                                  video_transform=val_transform,
                                                  num_segments=args.num_seg,
                                                  gamma=args.gamma)

    print('{} samples found, {} train data and {} test data.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    print(train_loader)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        prec1,prec5,lossClassification = validate(val_loader, model, criterion, -1)
        return

    for epoch in range(startEpoch, args.epochs):
#        if learning_rate_index > max_learning_rate_decay_count:
#            break
#        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = 0.0
        lossClassification = 0
        if (epoch + 1) % args.save_freq == 0:
            prec1,prec5,lossClassification = validate(val_loader, model, criterion, epoch)
            writer.add_scalar('data/top1_validation', prec1, epoch)
            writer.add_scalar('data/top3_validation', prec5, epoch)
            writer.add_scalar('data/classification_loss_validation', lossClassification, epoch)
            scheduler.step(lossClassification)
        # remember best prec@1 and save checkpoint
        
        is_best = prec1 >= best_prec1
        best_prec1 = max(prec1, best_prec1)
#        best_in_existing_learning_rate = max(prec1, best_in_existing_learning_rate)
#        
#        if best_in_existing_learning_rate > prec1 + 1:
#            learning_rate_index = learning_rate_index 
#            best_in_existing_learning_rate = 0        

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            if is_best:
                print("Model son iyi olarak kaydedildi")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_name, saveLocation)
    
    checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'best_loss': best_loss,
        'optimizer' : optimizer.state_dict(),
    }, is_best, checkpoint_name, saveLocation)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

def build_model():
    #args.arch：dark_light
    model = models.__dict__[args.arch](num_classes=11, length=args.num_seg, both_flow=args.both_flow)
    
    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model)
    model = model.cuda()
    
    return model

def build_model_validate():
    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)
    model=models.__dict__[args.arch](num_classes=11, length=args.num_seg, both_flow=args.both_flow)
   
    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model) 

    model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval() 
    return model

def build_model_continue():
    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)
    model=models.__dict__[args.arch](num_classes=11, length=args.num_seg, both_flow=args.both_flow)
   
    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model) 
        
    model.load_state_dict(params['state_dict'])
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    optimizer.load_state_dict(params['optimizer'])
    
    startEpoch = params['epoch']
    best_prec = params['best_prec1']
    return model, startEpoch, optimizer, best_prec


#进入
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch_classification = 0.0
    acc_mini_batch = 0.0
    acc_mini_batch_top3 = 0.0
    totalSamplePerIter=0
    for i, (inputs, inputs_light, targets) in enumerate(train_loader):
        inputs=inputs.view(-1,length,3,input_size,input_size).transpose(1,2)
        inputs_light=inputs_light.view(-1,length,3,input_size,input_size).transpose(1,2)

        inputs = inputs.cuda()
        inputs_light = inputs_light.cuda()

        targets = targets.cuda()
        output= model((inputs,inputs_light))

        prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
        acc_mini_batch += prec1.item()
        acc_mini_batch_top3 += prec5.item()

        lossClassification = criterion(output, targets)
        lossClassification = lossClassification / args.iter_size

        totalLoss=lossClassification
        loss_mini_batch_classification += lossClassification.data.item()
        totalLoss.backward()
        totalSamplePerIter +=  output.size(0)
        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()
            lossesClassification.update(loss_mini_batch_classification, totalSamplePerIter)
            top1.update(acc_mini_batch/args.iter_size, totalSamplePerIter)
            top5.update(acc_mini_batch_top3/args.iter_size, totalSamplePerIter)
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch_classification = 0
            acc_mini_batch = 0
            acc_mini_batch_top3 = 0.0
            totalSamplePerIter = 0.0
            #scheduler.step()
            
        if (i+1) % args.print_freq == 0:
            print('[%d] time: %.3f loss: %.4f' %(i,batch_time.avg,lossesClassification.avg))
          
    print('train * Epoch: {epoch} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'
          .format(epoch = epoch, top1=top1, top5=top5, lossClassification=lossesClassification))
    with open('train_record_%s.csv' % suffix, 'a', newline='') as f:
        record = csv.writer(f)
        record.writerow([epoch, round(top1.avg, 3), round(top5.avg, 3), round(lossesClassification.avg, 4)])
    writer.add_scalar('data/classification_loss_training', lossesClassification.avg, epoch)
    writer.add_scalar('data/top1_training', top1.avg, epoch)
    writer.add_scalar('data/top3_training', top5.avg, epoch)
def validate(val_loader, model, criterion,epoch):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inputs, inputs_light, targets) in enumerate(val_loader):
            inputs=inputs.view(-1,length,3,input_size,input_size).transpose(1,2)
            inputs_light=inputs_light.view(-1,length,3,input_size,input_size).transpose(1,2)

            inputs = inputs.cuda()
            inputs_light = inputs_light.cuda()
            targets = targets.cuda()
    
            # compute output
            output= model((inputs,inputs_light))
            
            lossClassification = criterion(output, targets)
    
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
            
            lossesClassification.update(lossClassification.data.item(), output.size(0))
            
            top1.update(prec1.item(), output.size(0))
            top5.update(prec5.item(), output.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    
        print('validate * * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'
              .format(top1=top1, top5=top5, lossClassification=lossesClassification))
        with open('validate_record_%s.csv' % suffix, 'a', newline='') as f:
            record = csv.writer(f)
            record.writerow([epoch, round(top1.avg,3), round(top5.avg,3), round(lossesClassification.avg,4)])
    return top1.avg, top5.avg, lossesClassification.avg

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(cur_path, best_path)

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
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    lr = args.lr * decay
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def adjust_learning_rate2(optimizer, epoch):
    isWarmUp=epoch < warmUpEpoch
    decayRate=0.2
    if isWarmUp:
        lr=args.lr*(epoch+1)/warmUpEpoch
    else:
        lr=args.lr*(1/(1+(epoch+1-warmUpEpoch)*decayRate))
    
    #decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def adjust_learning_rate3(optimizer, epoch):
    isWarmUp=epoch < warmUpEpoch
    decayRate=0.97
    if isWarmUp:
        lr=args.lr*(epoch+1)/warmUpEpoch
    else:
        lr = args.lr * decayRate**(epoch+1-warmUpEpoch)
    
    #decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def adjust_learning_rate4(optimizer, learning_rate_index):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** learning_rate_index
    lr = args.lr * decay
    print("Current learning rate is %4.8f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
