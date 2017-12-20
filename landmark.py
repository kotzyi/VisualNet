import torch
import time
from torch.autograd import Variable
import utils.utility as utility

def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    AM = []
    for i in range(9):
        AM.append(utility.AverageMeter())

    batch_time = AM[0]
    data_time = AM[1]
    losses = AM[2]
    lc_distance = AM[3]
    rc_distance = AM[4]
    ls_distance = AM[5]
    rs_distance = AM[6]
    lh_distance = AM[7]
    rh_distance = AM[8]

    model.train()
    end = time.time()

    for i, (input, collar, sleeve, hem, path) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_var = Variable(input)
        output = model(input_var)

        # Answers
        target = Variable(torch.cat((collar[:,2:6],sleeve[:,2:6],hem[:,2:6]),1)).float().cuda(async=True)
        target_v = torch.cat((collar[:,0:2],sleeve[:,0:2],hem[:,0:2]),1).long().cuda(async=True)

        lc_indices = Variable(torch.nonzero((0 == target_v[:,0]))[:,0]).cuda(async=True)
        rc_indices = Variable(torch.nonzero((0 == target_v[:,1]))[:,0]).cuda(async=True)

        ls_indices = Variable(torch.nonzero((0 == target_v[:,2]))[:,0]).cuda(async=True)
        rs_indices = Variable(torch.nonzero((0 == target_v[:,3]))[:,0]).cuda(async=True)

        lh_indices = Variable(torch.nonzero((0 == target_v[:,4]))[:,0]).cuda(async=True)
        rh_indices = Variable(torch.nonzero((0 == target_v[:,5]))[:,0]).cuda(async=True)

        lc_loss = criterion(output[:,0:2].index_select(0,lc_indices), target[:,0:2].index_select(0,lc_indices))
        rc_loss = criterion(output[:,2:4].index_select(0,rc_indices), target[:,2:4].index_select(0,rc_indices))

        ls_loss = criterion(output[:,4:6].index_select(0,ls_indices), target[:,4:6].index_select(0,ls_indices))
        rs_loss = criterion(output[:,6:8].index_select(0,rs_indices), target[:,6:8].index_select(0,rs_indices))

        lh_loss = criterion(output[:,12:14].index_select(0,lh_indices), target[:,8:10].index_select(0,lh_indices))
        rh_loss = criterion(output[:,14:16].index_select(0,rh_indices), target[:,10:12].index_select(0,rh_indices))

        loss = lh_loss + rh_loss + lc_loss + rc_loss + ls_loss + rs_loss
        losses.update(loss.data[0], input.size(0))

        lc_dist = utility.distance(output[:,0:2].index_select(0,lc_indices),target[:,0:2].index_select(0,lc_indices))
        rc_dist = utility.distance(output[:,2:4].index_select(0,rc_indices),target[:,2:4].index_select(0,rc_indices))
        ls_dist = utility.distance(output[:,4:6].index_select(0,ls_indices),target[:,4:6].index_select(0,ls_indices))
        rs_dist = utility.distance(output[:,6:8].index_select(0,rs_indices),target[:,6:8].index_select(0,rs_indices))
        lh_dist = utility.distance(output[:,12:14].index_select(0,lh_indices),target[:,8:10].index_select(0,lh_indices))
        rh_dist = utility.distance(output[:,14:16].index_select(0,rh_indices),target[:,10:12].index_select(0,rh_indices))

        lc_distance.update(lc_dist.data[0], 1)
        rc_distance.update(rc_dist.data[0], 1)
        ls_distance.update(ls_dist.data[0], 1)
        rs_distance.update(rs_dist.data[0], 1)
        lh_distance.update(lh_dist.data[0], 1)
        rh_distance.update(rh_dist.data[0], 1)

        #compute gradient and do SGD step
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
                'Location Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'LC_Dists {lc.val:.3f} ({lc.avg:.3f})\t'
                'RC_Dists {rc.val:.3f} ({rc.avg:.3f})\t'
                'LS_Dists {ls.val:.3f} ({ls.avg:.3f})\t'
                'RS_Dists {rs.val:.3f} ({rs.avg:.3f})\t'
                'LH_Dists {lh.val:.3f} ({lh.avg:.3f})\t'
                'RH_Dists {rh.val:.3f} ({rh.avg:.3f})\t'
                .format(epoch, i, len(train_loader), batch_time=batch_time,data_time=data_time, loss = losses, lc = lc_distance, rc = rc_distance, ls = ls_distance, rs = rs_distance, lh = lh_distance, rh = rh_distance))

def validate(val_loader, model, print_freq):
    AM = []
    for i in range(9):
        AM.append(utility.AverageMeter())

    batch_time = AM[0]
    data_time = AM[1]
    lc_distance = AM[2]
    rc_distance = AM[3]
    ls_distance = AM[4]
    rs_distance = AM[5]
    lh_distance = AM[6]
    rh_distance = AM[7]
    all_distance = AM[8]
    model.eval()
    end = time.time()

    for i, (input, collar, sleeve, hem, path) in enumerate(val_loader):
        data_time.update(time.time() - end)

        input_var = Variable(input, volatile=True)
        output = model(input_var).data

        # Answers
        target = torch.cat((collar[:,2:6],sleeve[:,2:6],hem[:,2:6]),1).float().cuda(async=True)
        target_v = torch.cat((collar[:,0:2],sleeve[:,0:2],hem[:,0:2]),1).long().cuda(async=True)

        lc_indices = torch.nonzero((0 == target_v[:,0]))[:,0].cuda(async=True)
        rc_indices = torch.nonzero((0 == target_v[:,1]))[:,0].cuda(async=True)

        ls_indices = torch.nonzero((0 == target_v[:,2]))[:,0].cuda(async=True)
        rs_indices = torch.nonzero((0 == target_v[:,3]))[:,0].cuda(async=True)

        lh_indices = torch.nonzero((0 == target_v[:,4]))[:,0].cuda(async=True)
        rh_indices = torch.nonzero((0 == target_v[:,5]))[:,0].cuda(async=True)

        lc_dist = utility.distance(output[:,0:2].index_select(0,lc_indices),target[:,0:2].index_select(0,lc_indices))
        rc_dist = utility.distance(output[:,2:4].index_select(0,rc_indices),target[:,2:4].index_select(0,rc_indices))
        ls_dist = utility.distance(output[:,4:6].index_select(0,ls_indices),target[:,4:6].index_select(0,ls_indices))
        rs_dist = utility.distance(output[:,6:8].index_select(0,rs_indices),target[:,6:8].index_select(0,rs_indices))
        lh_dist = utility.distance(output[:,12:14].index_select(0,lh_indices),target[:,8:10].index_select(0,lh_indices))
        rh_dist = utility.distance(output[:,14:16].index_select(0,rh_indices),target[:,10:12].index_select(0,rh_indices))

        lc_distance.update(lc_dist, 1)
        rc_distance.update(rc_dist, 1)
        ls_distance.update(ls_dist, 1)
        rs_distance.update(rs_dist, 1)
        lh_distance.update(lh_dist, 1)
        rh_distance.update(rh_dist, 1)
        all_distance.update(lc_dist+rc_dist+ls_dist+rs_dist+lh_dist+rh_dist,1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'LC_Dists {lc.val:.3f} ({lc.avg:.3f})\t'
                    'RC_Dists {rc.val:.3f} ({rc.avg:.3f})\t'
                    'LS_Dists {ls.val:.3f} ({ls.avg:.3f})\t'
                    'RS_Dists {rs.val:.3f} ({rs.avg:.3f})\t'
                    'LH_Dists {lh.val:.3f} ({lh.avg:.3f})\t'
                    'RH_Dists {rh.val:.3f} ({rh.avg:.3f})\t'
                    'Distance {all.val:.3f} ({all.avg:.3f})'
                    .format(i, len(val_loader), batch_time=batch_time,data_time=data_time,  lc = lc_distance, rc = rc_distance, ls = ls_distance, rs = rs_distance, lh = lh_distance, rh = rh_distance, all = all_distance))


    print('Test Result: Distances ({all.avg:.4f})'.format(all = all_distance))
    return all_distance.avg

