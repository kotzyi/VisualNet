import torch
import time
from torch.autograd import Variable
import utils.utility as utility

def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    AM = []
    for i in range(4):
        AM.append(utility.AverageMeter())

    batch_time = AM[0]
    data_time = AM[1]
    losses = AM[2]
    top1 = AM[3]

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = Variable(input).cuda(async=True)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 1))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, print_freq):
    AM = []
    for i in range(2):
        AM.append(utility.AverageMeter())

    batch_time = AM[0]
    top1 = AM[1]

    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = Variable(input, volatile=True).cuda(async=True)
        target_var = Variable(target, volatile=True)

        # compute output
        output = model(input_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 1))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #testing

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec {top1.val:.3f} ({top1.avg:.3f})\t'
                   i, len(val_loader), batch_time=batch_time, 
                   top1=top1))

    print(' * Prec {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

