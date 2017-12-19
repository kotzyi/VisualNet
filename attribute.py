import torch
import time
import re
from torch.autograd import Variable
import utils.utility as utility

W = 4 #weight
p = re.compile("-1|0|1")
cat_weight, att_weight = utility.get_weight()
att_weight = torch.FloatTensor(att_weight).cuda(async=True)

def train(train_loader, model, criterions, optimizer, epoch, print_freq):
    AM = []
    for i in range(8):
        AM.append(utility.AverageMeter())

    batch_time = AM[0]
    data_time = AM[1]
    att_losses = AM[2]
    cat_losses = AM[3]
    f1 = AM[4]
    prec = AM[5]
    recall = AM[6]
    accuracy = AM[7]

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, category,target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = [list(map(int, p.findall(x))) for x in target]
        target = torch.Tensor(target).gt(0)
        target = target.float().cuda(async=True)
        category = category.cuda(async=True)
        weight = torch.add(torch.mul(target*att_weight,W),4)
        input_var = Variable(input).cuda(async=True)
        target_var = Variable(target)

        category_var = Variable(category)

        # compute output
        _, att_output, cat_output = model(input_var)

        #attribute loss and f1 score
        criterions[0].weight = weight
        att_loss = criterions[0](att_output, target_var)
        f1_score, pr, re = utility.f1_score(att_output,target)

        #category loss and accuracy
        cat_weight = torch.FloatTensor(utility.cat_weight).cuda(async=True)
        criterions[1].weight = cat_weight
        cat_loss = criterions[1](cat_output, category_var) *0.1

        acc, _ = utility.accuracy(cat_output.data, category, topk=(1, 1))

        # measure accuracy and record loss of attribute
        att_losses.update(att_loss.data[0], input.size(0))
        f1.update(f1_score, input.size(0))
        prec.update(pr,input.size(0))
        recall.update(re,input.size(0))

        # measure accryacy and record loss of category
        cat_losses.update(cat_loss.data[0], input.size(0))
        accuracy.update(acc[0],input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        torch.autograd.backward([att_loss, cat_loss], [torch.ones(1).cuda(async=True), torch.ones(1).cuda(async=True)])
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Attribute Loss {att_loss.val:.4f} ({att_loss.avg:.4f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
				  'Category Loss {cat_loss.val:.4f} ({cat_loss.avg:.4f})\t'
				  'Accuracy {acc.val:.4f} ({acc.avg:.4f})'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, f1=f1, att_loss=att_losses,prec=prec,recall=recall,cat_loss=cat_losses,acc = accuracy))


def validate(val_loader, model, print_freq):
    AM =[]
    for i in range(5):
        AM.append(utility.AverageMeter())

    batch_time = AM[0]
    f1 = AM[1]
    prec = AM[2]
    recall = AM[3]
    accuracy = AM[4]

    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (input, category, target) in enumerate(val_loader):
        target = [list(map(int, p.findall(x))) for x in target]
        target = torch.Tensor(target).gt(0)
        target = target.float().cuda(async=True)
        category = category.cuda(async=True)
        input_var = Variable(input, volatile=True).cuda(async=True)
        target_var = Variable(target, volatile=True)
        category_var = Variable(category, volatile=True)

        # compute output
        _,att_output,cat_output = model(input_var)

        #attribute loss and f1 score
        f1_score, pr, re = utility.f1_score(att_output,target)

        #category loss and accuracy
        acc, _ = utility.accuracy(cat_output.data, category, topk=(1, 1))
        # measure accuracy and record loss of attribute
        f1.update(f1_score, input.size(0))
        prec.update(pr,input.size(0))
        recall.update(re,input.size(0))

        # measure accryacy and record loss of category
        accuracy.update(acc[0],input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #testing

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.4f})'
                  .format(i, len(val_loader), batch_time=batch_time, f1=f1,prec=prec,recall=recall,acc=accuracy))

    print(' * F1: {f1.avg:.3f} Accuracy: {acc.avg:.3f}'.format(f1=f1, acc=accuracy))

    return f1.avg

