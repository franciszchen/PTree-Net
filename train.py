import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import os
import random
import math
import json
import cv2

import config
import sys
sys.path.append("../models/")


def aug_np(nparray):
    """"nparray should be in range of [0, 1]"""
    RANDOM_BRIGHTNESS = 7
    RANDOM_CONTRAST = 5
    for i in range(nparray.shape[0]):
        # random flip
        if random.uniform(0, 1) < 0.5:
            nparray[i] = np.fliplr(nparray[i])    
        if random.uniform(0, 1) < 0.5:
            nparray[i] = np.flipud(nparray[i])    
        # random rotation
        r = random.randint(0, 3)
        if r:
            nparray[i] = np.rot90(nparray[i], r)
        # color jitter
        br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
        nparray[i] = nparray[i] + br
        # Random contrast
        cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
        nparray[i] = nparray[i] * cr
        # clip values to 0-1 range
        nparray[i] = np.clip(nparray[i], 0, 1.0)
    return nparray

def returnFAM(feature_conv, weight_softmax, class_idx, foremask):
    """check feature_conv>0, following relu
    """
    bz, nc, h, w = feature_conv.shape
    foremask = cv2.resize(np.squeeze(foremask, axis=0), (w, h), interpolation=cv2.INTER_NEAREST)
    output_cam = []
    cams_max_list = []
    cams_min_list = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        output_cam.append(cam)
        cams_max_list.append(np.max(cam))
        cams_min_list.append(np.min(cam))
    # calculate min&max in foreground
    cams_max = max(cams_max_list)
    cams_min = min(cams_min_list)

    output_cam_npy = np.stack(output_cam, axis=0)
    fam = np.zeros((h, w))
    for hi in range(h):
        for wj in range(w):
            fam[hi, wj] = np.std(output_cam_npy[:, hi, wj]) # std>=0
    #####
    fam = fam * foremask
    fam = (fam-np.min(fam))/(np.max(fam)-np.min(fam)+1e-10) 
    for idx in class_idx:
        output_cam[idx] = (output_cam[idx]-cams_min)/(cams_max-cams_min+1e-10) 
    return output_cam, fam


def get_fam(
    net, 
    features_blobs, 
    width,
    height,
    imgname, 
    foremask,
    fam_dir=None, 
    predict=None
    ):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    CAMs, fam_fs = returnFAM(features_blobs[0], weight_softmax, [i for i in range(weight_softmax.shape[0])], foremask)

    CAMs_list = []
    for i in range(len(CAMs)):
        CAM = cv2.resize(CAMs[i], (width, height), interpolation=cv2.INTER_NEAREST)
        CAMs_list.append(CAM)
    fam = cv2.resize(fam_fs, (width, height), interpolation=cv2.INTER_NEAREST)
    return CAMs_list, fam, fam_fs

def print_cz(str, f=None):
    if f is not None:
        print(str, file=f)
        if random.randint(0, 20) < 3:
            f.flush()
    print(str)

def time_mark():
    time_now = int(time.time())
    time_local = time.localtime(time_now)
    dt = time.strftime('%Y%m%d-%H%M%S', time_local)
    return(dt)

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def model_snapshot(model, new_file, old_file=None, save_dir='./', verbose=True, log_file=None):
    """
    :param model: network model to be saved
    :param new_file: new pth name
    :param old_file: old pth name
    :param verbose: more info or not
    :return: None
    """
    from collections import OrderedDict
    import torch

    if os.path.exists(save_dir) is False:
        os.makedirs(expand_user(save_dir))
        print_cz(str='Make new dir:'+save_dir, f=log_file)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    for file in os.listdir(save_dir):
        if old_file in file:
            if verbose:
                print_cz(str="Removing old model  {}".format(expand_user(save_dir + file)), f=log_file)
            os.remove(save_dir + file) # 
    if verbose:
        print_cz(str="Saving new model to {}".format(expand_user(save_dir + new_file)), f=log_file)
    time.sleep(5)
    torch.save(model.cpu().state_dict(),expand_user(save_dir + 'dict_'+new_file))
    time.sleep(5)
    model.cuda()

def prepare():
    """
        config, make dirs
    """
    args = config.get_args()
    time_tag = time_mark()
    log_dir = config.save_dir + time_tag + '_' + args.theme + '_' + args.optim + '_lr' + str(args.lr) + '_wd'+str(args.weight_decay) +\
        '_softbs'+str(args.soft_batch_size) + '_epochs'+str(args.epochs) + '_step'+str(args.lr_step) +'_gamma'+str(args.lr_gamma)

    if os.path.exists(log_dir) is False:# make dir if not exist
        os.makedirs(expand_user(log_dir))
        print('make dir: ' + str(log_dir))
    return args,  log_dir

def adjust_learning_rate(optimizer, lr, epoch, lr_step=40, lr_gamma=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (lr_gamma ** (epoch // lr_step)) # cuhk vgg16 setting
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0# value = current value
        self.avg = 0
        self.sum = 0# weighted sum
        self.count = 0# total sample num

    def update(self, value, n=1):# 
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # output, target -> FloatTensor, LongTensor
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)#keep maxk predictions
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))#expand target

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))# res is accuracy of current batch
    return res

def train(
    train_loader, 
    test_loader, 
    extractor_fam,
    extractor_tw,
    extractor_mid,
    extractor_bottom,
    aggregator,
    criterion, 
    criterion_sparse,
    criterion_semantic,
    optimizer_extractor_fam,
    optimizer_extractor_tw, 
    optimizer_extractor_mid, 
    optimizer_extractor_bottom,
    optimizer_aggregator,
    lr, 
    lr_fam_factor,
    lr_extractor_factor,
    lr_agg_factor,
    epochs, 
    soft_batch_size, 
    lr_step, 
    lr_gamma,
    patch_limits,
    logfile=None, 
    save_path=None, 
    start_epoch=0, 
    pth_prefix='', 
    csv_flag = False, 
    ):

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer_extractor_fam, lr*lr_fam_factor, epoch, lr_step, lr_gamma)# fam lr调整
        adjust_learning_rate(optimizer_extractor_tw, lr*lr_extractor_factor, epoch, lr_step, lr_gamma)# tw lr调整
        adjust_learning_rate(optimizer_extractor_mid, lr*lr_extractor_factor, epoch, lr_step, lr_gamma)# mid lr调整
        adjust_learning_rate(optimizer_extractor_bottom, lr*lr_extractor_factor, epoch, lr_step, lr_gamma)# bottom lr调整
        adjust_learning_rate(optimizer_aggregator, lr*lr_agg_factor, epoch, lr_step, lr_gamma) # agg的lr调整
        print_cz(str='Epoch:\t{:d}\t agg_lr:{:e}\t fam_lr:{:e}\t tw_lr:{:e}\t mid_lr:{:e}\t bottom_lr:{:e}'.format(epoch, optimizer_aggregator.param_groups[0]['lr'], 
            optimizer_extractor_fam.param_groups[0]['lr'],
            optimizer_extractor_tw.param_groups[0]['lr'],
            optimizer_extractor_mid.param_groups[0]['lr'],
            optimizer_extractor_bottom.param_groups[0]['lr'],
            ), f=logfile)
        train_top1, train_loss, train_kappa, [train_kappa_gcn, train_kappa_tw, train_kappa_mids, train_kappa_bottoms] = train_a_epoch(
            train_loader=train_loader, 
            extractor_fam=extractor_fam,
            extractor_tw=extractor_tw, 
            extractor_mid=extractor_mid, 
            extractor_bottom=extractor_bottom, 
            aggregator=aggregator,
            criterion=criterion, 
            criterion_sparse=criterion_sparse,
            criterion_semantic=criterion_semantic,
            optimizer_extractor_fam=optimizer_extractor_fam, 
            optimizer_extractor_tw=optimizer_extractor_tw, 
            optimizer_extractor_mid=optimizer_extractor_mid, 
            optimizer_extractor_bottom=optimizer_extractor_bottom, 
            optimizer_aggregator=optimizer_aggregator, 
            patch_limits=patch_limits,
            epoch=epoch, 
            soft_batch_size=soft_batch_size, 
            logfile=logfile)

        print_cz(" ==> Test ", f=logfile)
        test_top1, test_loss, test_f1, test_kappa, test_mcc, [test_kappa_gcn, test_kappa_tw, test_kappa_mids, test_kappa_bottoms] = test(
            test_loader=test_loader,
            extractor_fam=extractor_fam,
            extractor_tw=extractor_tw, 
            extractor_mid=extractor_mid,   
            extractor_bottom=extractor_bottom,
            aggregator=aggregator,
            criterion=criterion, 
            criterion_sparse=criterion_sparse,
            criterion_semantic=criterion_semantic,
            # soft_batch_size=soft_batch_size,
            patch_limits=patch_limits,
            epoch=epoch, 
            logfile=logfile)
        print_cz(" ", f=logfile)

        if epoch + 10 > epochs:
            model_snapshot(extractor_fam, new_file=(
                    pth_prefix+'extractorfam-last-{}-acc{:.3f}-f{:.3f}-kappa{:.3f}-mcc{:.3f}-gcn{:.1f}-tw{:.1f}-mids{:.1f}-bottoms{:.1f}-{}.pth'.format(epoch, test_top1, test_f1, test_kappa, test_mcc, test_kappa_gcn, test_kappa_tw, test_kappa_mids, test_kappa_bottoms, time_mark())
                    ), old_file=pth_prefix + 'extractor-no-remove-', save_dir=save_path , verbose=True) # 
            model_snapshot(extractor_tw, new_file=(
                    pth_prefix+'extractortw-last-{}-acc{:.3f}-f{:.3f}-kappa{:.3f}-mcc{:.3f}-gcn{:.1f}-tw{:.1f}-mids{:.1f}-bottoms{:.1f}-{}.pth'.format(epoch, test_top1, test_f1, test_kappa, test_mcc, test_kappa_gcn, test_kappa_tw, test_kappa_mids, test_kappa_bottoms, time_mark())
                    ), old_file=pth_prefix + 'extractor-no-remove-', save_dir=save_path , verbose=True) # 
            model_snapshot(extractor_mid, new_file=(
                    pth_prefix+'extractormid-last-{}-acc{:.3f}-f{:.3f}-kappa{:.3f}-mcc{:.3f}-gcn{:.1f}-tw{:.1f}-mids{:.1f}-bottoms{:.1f}-{}.pth'.format(epoch, test_top1, test_f1, test_kappa, test_mcc, test_kappa_gcn, test_kappa_tw, test_kappa_mids, test_kappa_bottoms, time_mark())
                    ), old_file=pth_prefix + 'extractor-no-remove-', save_dir=save_path , verbose=True) # 
            model_snapshot(extractor_bottom, new_file=(
                    pth_prefix+'extractorbottom-last-{}-acc{:.3f}-f{:.3f}-kappa{:.3f}-mcc{:.3f}-gcn{:.1f}-tw{:.1f}-mids{:.1f}-bottoms{:.1f}-{}.pth'.format(epoch, test_top1, test_f1, test_kappa, test_mcc, test_kappa_gcn, test_kappa_tw, test_kappa_mids, test_kappa_bottoms, time_mark())
                    ), old_file=pth_prefix + 'extractor-no-remove-', save_dir=save_path , verbose=True) # 
            model_snapshot(aggregator, new_file=(
                    pth_prefix+'aggregator-last-{}-acc{:.3f}-f{:.3f}-kappa{:.3f}-mcc{:.3f}-gcn{:.1f}-tw{:.1f}-mids{:.1f}-bottoms{:.1f}-{}.pth'.format(epoch, test_top1, test_f1, test_kappa, test_mcc,test_kappa_gcn, test_kappa_tw, test_kappa_mids, test_kappa_bottoms, time_mark())
                    ), old_file=pth_prefix + 'aggregator-no-remove-', save_dir=save_path , verbose=True) # 


def train_a_epoch(train_loader, 
    extractor_fam,
    extractor_tw,
    extractor_mid,
    extractor_bottom, 
    aggregator,
    criterion, 
    criterion_sparse,
    criterion_semantic,
    optimizer_extractor_fam,
    optimizer_extractor_tw, 
    optimizer_extractor_mid, 
    optimizer_extractor_bottom,
    optimizer_aggregator,
    patch_limits,
    epoch, 
    soft_batch_size,
    logfile=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_sparse = AverageMeter()
    losses_b2m = AverageMeter()
    losses_b2tw = AverageMeter()
    losses_m2tw = AverageMeter()
    losses_gcn2fam = AverageMeter()
    top1 = AverageMeter()

    extractor_fam.train()
    extractor_tw.train()
    extractor_mid.train()
    extractor_bottom.train()
    aggregator.train()
    epoch_start_time = time.time()
    end = time.time()

    extractor_fam.zero_grad()
    extractor_tw.zero_grad()
    extractor_mid.zero_grad()
    extractor_bottom.zero_grad()
    aggregator.zero_grad()
    
    pred_fam_list = []
    pred_list = []
    pred_gcn_list = []
    pred_tw_list = []
    pred_mids_list = []
    pred_bottoms_list = []
    label_list = []

    to_update_sample_count = 0 

    for idx, (wsi_str_tensor, label, tip, wsi_name_tensor, foremask_tensor) in enumerate(train_loader):
        wsi_handle = crop_patch_online.get_wsi_handle(wsi_str_tensor[0])
        wsi_name = wsi_name_tensor[0]
        
        label_batch = Variable(label).long().cuda()
        tip_batch = Variable(tip*foremask_tensor).cuda()
        _, _, tip_h, tip_w = tip_batch.shape

        # tip heatmap
        global features_blobs
        features_blobs = []
        output_tip_fam, _, fam_vector = extractor_fam(x=tip_batch)
        _, feature_tip_whole, _ = extractor_tw(x=tip_batch)
        _, fam_map, fam_fs_map = get_fam(net=extractor_fam, 
                features_blobs=features_blobs, 
                width=tip_w,
                height=tip_h, 
                imgname=wsi_name,
                foremask=torch.squeeze(foremask_tensor, dim=0).numpy(), 
                fam_dir=None)

        locationC_tip_fs_list, tip_patch_ratio = heatmap_issue.find_top_focal(
            heatmap=torch.from_numpy(fam_fs_map).cpu(), 
            ratio=0.1, 
            attention_patch_size=int(config.patch_size/4*(fam_fs_map.shape[1]/fam_map.shape[1])), # mid is 4x of tip whole input
            mask_ratio=1,
            patch_num_limit=patch_limits) #np loc
        factor = (fam_map.shape[0]+fam_map.shape[1])/(fam_fs_map.shape[0]+fam_fs_map.shape[1]) # between fam_fs and tip whole-tw

        ##### shuffle tip/mid loc
        random.shuffle(locationC_tip_fs_list)

        ####### map loc_fs to loc #######
        locationC_tip_list = [] # location in tip whole-tw
        for (loc0_fs, loc1_fs) in locationC_tip_fs_list:
            loc0 = int((loc0_fs+1)*factor - 1)
            loc1 = int((loc1_fs+1)*factor - 1)
            locationC_tip_list.append((loc0, loc1))
        
        #########

        # S means slides in openslide 0-level crop-region
        #### tip locs at Bottom        
        locationCS_tip_list = crop_patch_online.slide_loc_list(
            location_orig_list=locationC_tip_list, level_orig='tip') # np loc
        
        #### mid locs at Bottom
        psize_mid = config.patch_size
        psizeS_mid = psize_mid*int(config.scale_dict_checked['slide']/config.scale_dict_checked['mid']) # should be 2x
        locationCS_mid_list = locationCS_tip_list ####
       
        psize_bottom = config.patch_size
        psizeS_bottom = psize_bottom*int(config.scale_dict_checked['slide']/config.scale_dict_checked['bottom']) # should be 1x
        locationCS_bottom_list2 = []
        for mid_loc_idx in range(len(locationCS_mid_list)):
            locationCS_bottom_list = crop_patch_online.patchC_loc_list(
                locationCS_mid_list[mid_loc_idx], 
                psizeS=psizeS_bottom)
            #### shuffle 4 bottom loc in each mid loc
            random.shuffle(locationCS_bottom_list)
            locationCS_bottom_list2.append(locationCS_bottom_list)
       
        """
        for the mid level
        crop multi-patch at mid-level
        tip-level serves as a single node (root node) of ptree 
        """
        patch_mid_list, locationCSR_mid_list = crop_patch_online.crop_locationCS_patches(
            wsi_handle=wsi_handle, 
            level=config.scale_openslide_dict_checked['mid'],
            locationCS_list=locationCS_mid_list, 
            psizeS=psizeS_mid,
            psize=psize_mid) # list
        np_mid = aug_np(
            np.stack(patch_mid_list, axis=0).astype(np.float32)/255.0
        )
        input_mid = Variable(
            torch.FloatTensor(
                np.transpose(np_mid, (0, 3, 1, 2))
                ).cuda()
            ) 
  
        """
        for the bottom level
        """
        tensor_bottom_list = [] # list of tensors
        for mid_idx in range(len(locationCS_bottom_list2)): 
            patch_bottom_list, locationCSR_bottom_list = crop_patch_online.crop_locationCS_patches(
                wsi_handle=wsi_handle, 
                level=config.scale_openslide_dict_checked['bottom'], 
                locationCS_list=locationCS_bottom_list2[mid_idx], 
                psizeS=psizeS_bottom,
                psize=psize_bottom) # 
            np_bottom = aug_np(
                np.stack(patch_bottom_list, axis=0).astype(np.float32)/255.0
                )
            tensor_bottom_list.append(
                torch.FloatTensor(
                    np.transpose(np_bottom, (0, 3, 1, 2))
                    )
                ) # list of tensors 
        input_bottom = Variable(
            torch.cat(tensor_bottom_list, dim=0).cuda()
            ) # tensor

        # data time
        data_time.update(time.time() - end)
        ### concat 3-level input tensor
        _, features_mid, _ = extractor_mid(x=input_mid) # mid level
        _, features_bottom, _ = extractor_bottom(x=input_bottom) # bottom level
        features_2scales = torch.cat(
                        [feature_tip_whole, features_mid, features_bottom], 
                        dim=0)
        ###

        matrix, edge_list, [mid_dim, bottom_dim] = graph_issue.create_matrix(bottom_list2=locationCS_bottom_list2)
        edge_list = graph_issue.pyg_edge_tensor(edge_list=edge_list).cuda()
        out_gcn, out_tw, out_mids, out_bottoms,\
            gcn_feature, tw_feature, mid_average_feature, bottom_weighted_feature, \
                bottom_weights_vector,\
                    mid_features, tree_bottom_features = aggregator(
            features_2scales, edge_index=edge_list, arch_list=[mid_dim, bottom_dim])
        # equation 9
        loss_tmp = args.gcn_ratio*criterion(out_gcn, label_batch) + criterion(out_tw, label_batch) + criterion(out_mids, label_batch) + criterion(out_bottoms, label_batch)
        # equation 6
        loss_sparse = args.sparse_ratio*criterion_sparse(
            input=bottom_weights_vector, 
            target=torch.zeros(bottom_weights_vector.shape).long().cuda())
        
        # equation 7
        loss_b2mid = 0
        for node_idx in range(mid_features.shape[0]):
            loss_b2mid = loss_b2mid + \
                args.bottom2mid_ratio * (
                criterion_semantic(input=F.normalize(input=mid_features[node_idx], p=2, dim=-1), target=F.normalize(input=tree_bottom_features[node_idx].clone().detach(), p=2, dim=-1)) +\
                criterion_semantic(input=F.normalize(input=tree_bottom_features[node_idx], p=2, dim=-1), target=F.normalize(input=mid_features[node_idx].clone().detach(), p=2, dim=-1)))/2
        loss_b2mid = loss_b2mid/ mid_features.shape[0]
        # equation 8
        loss_m2tw = args.mid2tw_ratio * (
            criterion_semantic(input=F.normalize(input=tw_feature, p=2, dim=-1), target=F.normalize(input=mid_average_feature.clone().detach(), p=2, dim=-1))+\
            criterion_semantic(input=F.normalize(input=mid_average_feature, p=2, dim=-1), target=F.normalize(input=tw_feature.clone().detach(), p=2, dim=-1)))/2

        loss_gcn2fam = args.gcn2fam_ratio*criterion_semantic(input=F.normalize(input=fam_vector, p=2, dim=-1), target=F.normalize(input=gcn_feature.clone().detach(), p=2, dim=-1))

        # equation 10
        loss = loss_tmp + loss_sparse + loss_b2mid + loss_m2tw
        
        losses.update(loss.data.item(), 1)
        losses_sparse.update(loss_sparse.data.item(), 1)
        losses_b2m.update(loss_b2mid.data.item(), 1)
        losses_m2tw.update(loss_m2tw.data.item(), 1)
        losses_gcn2fam.update(loss_gcn2fam.item(), 1)

        loss_fam = criterion(output_tip_fam, label_batch) + loss_gcn2fam

        prec1 = accuracy(out_gcn.data, label_batch.data, topk=(1,))[0]        
        top1.update(prec1[0], 1)

        # for confusion matrix
        _, pred_fam = output_tip_fam.topk(1, 1, True, True)
        pred_fam_list.extend(
            ((pred_fam.cpu()).numpy()).tolist())

        _, pred_gcn = out_gcn.topk(1, 1, True, True)
        pred_gcn_list.extend(
            ((pred_gcn.cpu()).numpy()).tolist())
        _, pred_tw = out_tw.topk(1, 1, True, True)
        pred_tw_list.extend(
            ((pred_tw.cpu()).numpy()).tolist())
        _, pred_mids = out_mids.topk(1, 1, True, True)
        pred_mids_list.extend(
            ((pred_mids.cpu()).numpy()).tolist())
        _, pred_bottoms = out_bottoms.topk(1, 1, True, True)
        pred_bottoms_list.extend(
            ((pred_bottoms.cpu()).numpy()).tolist())
        
        # 
        pred_ens = pred_gcn
        pred_list.extend(
            ((pred_ens.cpu()).numpy()).tolist())
        label_list.extend(
            ((label.cpu()).numpy()).tolist())
        # equation 11
        (loss+loss_fam).backward()

        to_update_sample_count += 1

        if (idx+1) % soft_batch_size == 0 or (idx+1 == len(train_loader)):
            optimizer_extractor_fam.step()
            optimizer_extractor_tw.step()
            optimizer_extractor_mid.step()
            optimizer_extractor_bottom.step()
            optimizer_aggregator.step()
            
            if (idx+1) % (5*soft_batch_size) == 0:
                print_cz(' \t Batch {}, Train Loss {:.3f} SparseL {:.3f} b2mL {:.3f} b2twL {:.3f} m2twL {:.3f} gcn2famL {:.3f} Prec@1 {:.3f}'.\
                format(idx, losses.value, losses_sparse.value, losses_b2m.value, losses_b2tw.value, losses_m2tw.value, losses_gcn2fam.value, top1.value), f=logfile)
            if idx+1 == len(train_loader):
                print('final update of incomplete batch in this epoch: {:d}/{:d}'.format(to_update_sample_count, len(train_loader)%soft_batch_size))

            optimizer_extractor_fam.zero_grad()
            optimizer_extractor_tw.zero_grad()
            optimizer_extractor_mid.zero_grad()
            optimizer_extractor_bottom.zero_grad()
            optimizer_aggregator.zero_grad()
            to_update_sample_count = 0

        features_blobs = []
        del tip, label
        del tip_batch, input_mid, input_bottom, label_batch
        del fam_map, fam_fs_map
        del tensor_bottom_list
        del output_tip_fam
        del feature_tip_whole, features_bottom, features_mid, features_2scales
        del loss, loss_tmp
        del np_bottom, np_mid
        del gcn_feature, tw_feature, mid_average_feature
        del bottom_weighted_feature, bottom_weights_vector
        del fam_vector
        
        # batch time updated
        batch_time.update(time.time() - end)
        end = time.time()

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    print_cz(str=' * Train time {:.3f}\t  BatchT: {:.3f}\t DataT: {:.3f}\t D/B: {:.1f}%'.format(epoch_time, batch_time.avg, data_time.avg, 100.0*(data_time.avg/batch_time.avg)), f=logfile)# print top*.avg
    print_cz(str='   Loss {:.3f}\t\t TrainEpoch-{:d}'.format(losses.avg, epoch), f=logfile)# print top*.avg
    print_cz(str='SparseLoss{:.3f}'.format(losses_sparse.avg), f=logfile)
    print_cz(str='b2mLoss{:.3f}'.format(losses_b2m.avg), f=logfile)
    print_cz(str='m2twLoss{:.3f}'.format(losses_m2tw.avg), f=logfile)
    print_cz(str='gcn2famLoss{:.3f}'.format(losses_gcn2fam.avg), f=logfile)

    print_cz(str='---', f=logfile)
    print_cz(str=metrics.classification_report(y_true=label_list, y_pred=pred_list, digits=5), f=logfile)
    train_f1_macro = 100*metrics.f1_score(y_true=label_list, y_pred=pred_list, average='macro')
    train_mcc = 100*metrics.matthews_corrcoef(y_true=label_list, y_pred=pred_list)
    
    kappa_fam = 100*metrics.cohen_kappa_score(y1=label_list, y2=pred_fam_list, weights='quadratic')
    kappa = 100*metrics.cohen_kappa_score(y1=label_list, y2=pred_list, weights='quadratic')
    kappa_gcn = 100*metrics.cohen_kappa_score(y1=label_list, y2=pred_gcn_list, weights='quadratic')
    kappa_tw = 100*metrics.cohen_kappa_score(y1=label_list, y2=pred_tw_list, weights='quadratic')
    kappa_mids = 100*metrics.cohen_kappa_score(y1=label_list, y2=pred_mids_list, weights='quadratic')
    kappa_bottoms = 100*metrics.cohen_kappa_score(y1=label_list, y2=pred_bottoms_list, weights='quadratic')

    print_cz(str='Train Prec@1 {:.3f}%\t F1 {:.3f}%\t Kappa {:.3f}%\t mcc {:.3f}%\t\t TrainEpoch-{:d}'.format(top1.avg, train_f1_macro, kappa, train_mcc, epoch), f=logfile)# print top*.avg
    
    print_cz(str=metrics.confusion_matrix(y_true=label_list, y_pred=pred_list), f=logfile)

    print_cz('kappa:\t{:.3f}'.format(kappa), f=logfile)
    print_cz('-- kappa gcn :\t{:.3f}'.format(kappa_gcn), f=logfile)
    print_cz('-- kappa tw  :\t{:.3f}'.format(kappa_tw), f=logfile)
    print_cz('-- kappa mids:\t{:.3f}'.format(kappa_mids), f=logfile)
    print_cz('-- kappa bottoms:\t{:.3f}'.format(kappa_bottoms), f=logfile)
    print_cz('-- kappa fam:\t{:.3f}'.format(kappa_fam), f=logfile)
    print_cz('mcc:\t{:.3f}'.format(100*metrics.matthews_corrcoef(y_true=label_list, y_pred=pred_list)), f=logfile)
    print_cz(str='---', f=logfile)

    return top1.avg, losses.avg, kappa, [kappa_gcn, kappa_tw, kappa_mids, kappa_bottoms]


def test(test_loader,
    extractor_fam,
    extractor_tw,
    extractor_mid,  
    extractor_bottom,
    aggregator,
    criterion, 
    criterion_sparse,
    criterion_semantic,
    patch_limits,
    # batch_size, 
    epoch=0, 
    logfile=None):

    extractor_fam.cuda()
    extractor_tw.cuda()
    extractor_mid.cuda()
    extractor_bottom.cuda()
    extractor_fam.train()
    extractor_tw.train()
    extractor_mid.train()
    extractor_bottom.train()
    aggregator.cuda()
    aggregator.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_sparse = AverageMeter()
    losses_b2m = AverageMeter()
    losses_b2tw = AverageMeter()
    losses_m2tw = AverageMeter()
    losses_gcn2fam = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    test_start_time = time.time()

    pred_fam_list = []
    pred_list = []
    pred_gcn_list = []
    pred_tw_list = []
    pred_mids_list = []
    pred_bottoms_list = []
    label_list = []

    with torch.no_grad():
        for idx, (wsi_str_tensor, label, tip, wsi_name_tensor, foremask_tensor) in enumerate(test_loader):
            wsi_handle = crop_patch_online.get_wsi_handle(wsi_str_tensor[0])
            wsi_name = wsi_name_tensor[0]
            tip_batch = Variable(tip*foremask_tensor).cuda()
            label_batch = torch.autograd.Variable(label, volatile=True).long().cuda()

            _, _, tip_h, tip_w = tip_batch.shape
            # tip heatmap
            global features_blobs
            features_blobs = []
            output_tip_fam, _, fam_vector = extractor_fam(x=tip_batch)
            _, feature_tip_whole, _ = extractor_tw(x=tip_batch)
            _, fam_map, fam_fs_map = get_fam(net=extractor_fam, 
                    features_blobs=features_blobs, 
                    width=tip_w,
                    height=tip_h, 
                    imgname=wsi_name,
                    foremask=torch.squeeze(foremask_tensor, dim=0).numpy(), 
                    fam_dir=None)
            # get tip_focal from net and after mask
            locationC_tip_fs_list, tip_patch_ratio = heatmap_issue.find_top_focal(
                    heatmap=torch.from_numpy(fam_fs_map).cpu(), 
                    ratio=0.1, 
                    attention_patch_size=int(config.patch_size/4*(fam_fs_map.shape[1]/fam_map.shape[1])), 
                    mask_ratio=1,
                    patch_num_limit=patch_limits) #np loc
            factor = (fam_map.shape[0]+fam_map.shape[1])/(fam_fs_map.shape[0]+fam_fs_map.shape[1])

            ####### map loc_fs to loc #######
            locationC_tip_list = []
            for (loc0_fs, loc1_fs) in locationC_tip_fs_list:
                loc0 = int((loc0_fs+1)*factor - 1)
                loc1 = int((loc1_fs+1)*factor - 1)
                locationC_tip_list.append((loc0, loc1))
            #########

            # S means slides in openslide 0-level crop-region
            #### tip locs at Bottom
            locationCS_tip_list = crop_patch_online.slide_loc_list(
                location_orig_list=locationC_tip_list, level_orig='tip') # np loc
            #### mid locs at Bottom
            psize_mid = config.patch_size
            psizeS_mid = psize_mid*int(config.scale_dict_checked['slide']/config.scale_dict_checked['mid']) # should be 2x
            locationCS_mid_list = locationCS_tip_list ####

            psize_bottom = config.patch_size
            psizeS_bottom = psize_bottom*int(config.scale_dict_checked['slide']/config.scale_dict_checked['bottom']) # should be 1x
            locationCS_bottom_list2 = []
            for mid_loc_idx in range(len(locationCS_mid_list)):
                locationCS_bottom_list = crop_patch_online.patchC_loc_list(
                    locationCS_mid_list[mid_loc_idx], 
                    psizeS=psizeS_bottom) # one location produces 4 patches
                locationCS_bottom_list2.append(locationCS_bottom_list)
            #####
            """
            for the mid level
            """
            patch_mid_list, locationCSR_mid_list = crop_patch_online.crop_locationCS_patches(
                wsi_handle=wsi_handle, 
                level=config.scale_openslide_dict_checked['mid'],
                locationCS_list=locationCS_mid_list, 
                psizeS=psizeS_mid,
                psize=psize_mid
                ) # list
            np_mid = np.stack(patch_mid_list, axis=0).astype(np.float32)/255.0
            input_mid = Variable(
                torch.FloatTensor(
                    np.transpose(np_mid, (0, 3, 1, 2))
                    ).cuda()
            ) 
            """
            for the bottom level
            """
            tensor_bottom_list = [] # list of tensors
            for mid_idx in range(len(locationCS_bottom_list2)): 
                patch_bottom_list, locationCSR_bottom_list = crop_patch_online.crop_locationCS_patches(
                    wsi_handle=wsi_handle, 
                    level=config.scale_openslide_dict_checked['bottom'], 
                    locationCS_list=locationCS_bottom_list2[mid_idx], 
                    psizeS=psizeS_bottom,
                    psize=psize_bottom) # 
                np_bottom = np.stack(patch_bottom_list, axis=0).astype(np.float32)/255.0
                tensor_bottom_list.append(
                    torch.FloatTensor(
                        np.transpose(np_bottom, (0, 3, 1, 2))
                        )
                    ) # list of tensors 
            input_bottom = Variable(
                torch.cat(tensor_bottom_list, dim=0).cuda()
                ) # tensor
            
            # data time
            data_time.update(time.time() - end)
            ### concat 3-level features
            _, features_mid, _ = extractor_mid(x=input_mid) # mid level
            _, features_bottom, _ = extractor_bottom(x=input_bottom) # bottom level
            features_2scales = torch.cat(
                            [feature_tip_whole, features_mid, features_bottom], 
                            dim=0)
            
            #
            matrix, edge_list, [mid_dim, bottom_dim] = graph_issue.create_matrix(bottom_list2=locationCS_bottom_list2)
            edge_list = graph_issue.pyg_edge_tensor(edge_list=edge_list).cuda()
            out_gcn, out_tw, out_mids, out_bottoms, \
                gcn_feature, tw_feature, mid_average_feature, bottom_weighted_feature, bottom_weights_vector,\
                    mid_features, tree_bottom_features = aggregator(
                features_2scales, edge_index=edge_list, arch_list=[mid_dim, bottom_dim])
            loss_tmp = args.gcn_ratio*criterion(out_gcn, label_batch) + criterion(out_tw, label_batch) + criterion(out_mids, label_batch) + criterion(out_bottoms, label_batch) 
            loss_sparse = args.sparse_ratio*criterion_sparse(
                input=bottom_weights_vector, 
                target=torch.zeros(bottom_weights_vector.shape).long().cuda())
            loss_b2mid = 0
            for node_idx in range(mid_features.shape[0]):
                loss_b2mid = loss_b2mid + \
                    args.bottom2mid_ratio * (
                    criterion_semantic(input=F.normalize(input=mid_features[node_idx], p=2, dim=-1), target=F.normalize(input=tree_bottom_features[node_idx].clone().detach(), p=2, dim=-1)) +\
                    criterion_semantic(input=F.normalize(input=tree_bottom_features[node_idx], p=2, dim=-1), target=F.normalize(input=mid_features[node_idx].clone().detach(), p=2, dim=-1)))/2
            loss_b2mid = loss_b2mid/ mid_features.shape[0]
            loss_m2tw = args.mid2tw_ratio * (
                criterion_semantic(input=F.normalize(input=tw_feature, p=2, dim=-1), target=F.normalize(input=mid_average_feature.clone().detach(), p=2, dim=-1))+\
                criterion_semantic(input=F.normalize(input=mid_average_feature, p=2, dim=-1), target=F.normalize(input=tw_feature.clone().detach(), p=2, dim=-1)))/2

            loss_gcn2fam = args.gcn2fam_ratio*criterion_semantic(input=F.normalize(input=fam_vector, p=2, dim=-1), target=F.normalize(input=gcn_feature.clone().detach(), p=2, dim=-1))

            ###
            loss = loss_tmp + loss_sparse + loss_b2mid + loss_m2tw

            losses.update(loss.data.item(), 1) 
            losses_sparse.update(loss_sparse.data.item(), 1) 
            losses_b2m.update(loss_b2mid.data.item(), 1) 
            losses_m2tw.update(loss_m2tw.data.item(), 1)
            losses_gcn2fam.update(loss_gcn2fam.item(), 1)

            prec1 = accuracy(out_gcn.data, label_batch.data, topk=(1,))[0]        
            top1.update(prec1[0], 1)

            # for confusion matrix
            _, pred_fam = output_tip_fam.topk(1, 1, True, True)
            pred_fam_list.extend(
                ((pred_fam.cpu()).numpy()).tolist())
            
            _, pred_gcn = out_gcn.topk(1, 1, True, True)
            pred_gcn_list.extend(
                ((pred_gcn.cpu()).numpy()).tolist())
            _, pred_tw = out_tw.topk(1, 1, True, True)
            pred_tw_list.extend(
                ((pred_tw.cpu()).numpy()).tolist())
            _, pred_mids = out_mids.topk(1, 1, True, True)
            pred_mids_list.extend(
                ((pred_mids.cpu()).numpy()).tolist())
            _, pred_bottoms = out_bottoms.topk(1, 1, True, True)
            pred_bottoms_list.extend(
                ((pred_bottoms.cpu()).numpy()).tolist())
            # #
            pred_ens = pred_gcn
            pred_list.extend(
                ((pred_ens.cpu()).numpy()).tolist())

            label_list.extend(
                ((label.cpu()).numpy()).tolist())

            features_blobs = []
            del tip, label
            del tip_batch, input_mid, input_bottom, label_batch
            del fam_map, fam_fs_map
            del tensor_bottom_list

            del output_tip_fam
            del feature_tip_whole, features_mid, features_bottom, features_2scales
            del loss
            del np_mid, np_bottom
            del gcn_feature, tw_feature, mid_average_feature
            del bottom_weighted_feature, bottom_weights_vector
            del fam_vector
            # batch time updated
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 100 == 0:
                print_cz(' \t Batch {}, Test  Loss {:.3f} SparseL {:.3f} b2mL {:.3f} b2twL {:.3f} m2twL {:.3f} gcn2famL {:.3f} Prec@1 {:.3f}'.\
                    format(idx, losses.value, losses_sparse.value, losses_b2m.value, losses_b2tw.value, losses_m2tw.value, losses_gcn2fam.value, top1.value), f=logfile)

    test_end_time = time.time()# 
    test_time = test_end_time - test_start_time
    print_cz(str=' * Test  time {:.3f}\t  BatchT: {:.3f}\t DataT: {:.3f}\t D/B: {:.1f}%'.format(test_time, batch_time.avg, data_time.avg, 100.0*(data_time.avg/batch_time.avg)), f=logfile)# print top*.avg
    print_cz(str='   Loss {:.3f}\t\t TestEpoch-{:d}'.format(losses.avg, epoch), f=logfile)# print top*.avg
    print_cz(str='SparseLoss{:.3f}'.format(losses_sparse.avg), f=logfile)
    print_cz(str='b2mLoss{:.3f}'.format(losses_b2m.avg), f=logfile)
    print_cz(str='m2twLoss{:.3f}'.format(losses_m2tw.avg), f=logfile)
    print_cz(str='gcn2famLoss{:.3f}'.format(losses_gcn2fam.avg), f=logfile)

    test_f1_macro = 100*metrics.f1_score(y_true=label_list, y_pred=pred_list, average='macro')
    test_mcc = 100*metrics.matthews_corrcoef(y_true=label_list, y_pred=pred_list)

    kappa = 100*metrics.cohen_kappa_score(y1=label_list, y2=pred_list, weights='quadratic')
    kappa_fam = 100*metrics.cohen_kappa_score(y1=label_list, y2=pred_fam_list, weights='quadratic')
    kappa_gcn = 100*metrics.cohen_kappa_score(y1=label_list, y2=pred_gcn_list, weights='quadratic')
    kappa_tw = 100*metrics.cohen_kappa_score(y1=label_list, y2=pred_tw_list, weights='quadratic')
    kappa_mids = 100*metrics.cohen_kappa_score(y1=label_list, y2=pred_mids_list, weights='quadratic')
    kappa_bottoms = 100*metrics.cohen_kappa_score(y1=label_list, y2=pred_bottoms_list, weights='quadratic')

    print_cz(str='   Prec@1 {:.3f}%\t F1 {:.3f}%\t Kappa {:.3f}%\t mcc {:.3f}%\t\t TestEpoch-{:d}'.format(top1.avg, test_f1_macro, kappa, test_mcc, epoch), f=logfile)# print top*.avg

    print_cz(str='---', f=logfile)
    print_cz(str=metrics.classification_report(y_true=label_list, y_pred=pred_list, digits=5), f=logfile)
    print_cz(str=metrics.confusion_matrix(y_true=label_list, y_pred=pred_list), f=logfile)

    print_cz('kappa:\t{:.3f}'.format(kappa), f=logfile)
    print_cz('-- kappa gcn :\t{:.3f}'.format(kappa_gcn), f=logfile)
    print_cz('-- kappa tw  :\t{:.3f}'.format(kappa_tw), f=logfile)
    print_cz('-- kappa mids:\t{:.3f}'.format(kappa_mids), f=logfile)
    print_cz('-- kappa bottoms:\t{:.3f}'.format(kappa_bottoms), f=logfile)
    print_cz('-- kappa fam:\t{:.3f}'.format(kappa_fam), f=logfile)
    print_cz('mcc:\t{:.3f}'.format(test_mcc), f=logfile)
    print_cz(str='---', f=logfile)

    return top1.avg, losses.avg, test_f1_macro, kappa, test_mcc, [kappa_gcn, kappa_tw, kappa_mids, kappa_bottoms]

if __name__ == '__main__':
    from models import resnet_in_fc3_bypass
    from models import ptree_aggregator
    import dataloader
    import crop_patch_online
    import heatmap_issue
    import graph_issue

    from sklearn import metrics

    print('start...')

    args, log_dir = prepare()
    log_file = open((log_dir + '/' + 'print_out_screen.txt'), 'w')
    print_cz("===> Preparing", f=log_file)
    t = time.time()
    with open(log_dir + '/setting.json', 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))
        print_cz(json.dumps(args.__dict__, indent=4), f=log_file)

    print_cz("===> Building model", f=log_file)
    pth_dir = r'your pth dir'
    # pre-trained resnet-18 for different splits (blue network in Fig. 2)
    pth_folder_list = [
        r'resnet18-3class-0split/',
        r'resnet18-3class-1split/',
        r'resnet18-3class-2split/',
        r'resnet18-3class-3split/'
    ]
    pth_filename_list = [
        r'extractor.pth',
        r'extractor.pth',
        r'extractor.pth',
        r'extractor.pth'
    ]
    #####
    # extractor load pretrained weights
    # f(theta_FAM) in Fig. 2 
    extractor_fam = torch.load(pth_dir + pth_folder_list[args.test_split] + pth_filename_list[args.test_split])
    # f(theta_1) in Fig. 2 
    extractor_tw = torch.load(pth_dir + pth_folder_list[args.test_split] + pth_filename_list[args.test_split])
    # f(theta_2) in Fig. 2 
    extractor_mid = torch.load(pth_dir + pth_folder_list[args.test_split] + pth_filename_list[args.test_split])
    # f(theta_3) in Fig. 2 
    extractor_bottom = torch.load(pth_dir + pth_folder_list[args.test_split] + pth_filename_list[args.test_split])
    global features_blobs
    features_blobs = []
    def hook_feature(module, input, output):
        global features_blobs
        features_blobs.append(output.data.cpu().numpy())
    extractor_fam._modules.get('layer4').register_forward_hook(hook_feature)

    aggregator = ptree_aggregator.Aggregator(
        class_num=3, 
        in_dim=512, 
        inter_dim=128, 
        out_dim=64)
    criterion = nn.CrossEntropyLoss() # 
    criterion_sparse = nn.L1Loss(size_average=None)
    criterion_semantic = nn.MSELoss(size_average=None)

    print_cz("===> Setting GPU", f=log_file)
    extractor_fam = extractor_fam.cuda()
    extractor_tw = extractor_tw.cuda()
    extractor_mid = extractor_mid.cuda()
    extractor_bottom = extractor_bottom.cuda()
    aggregator = aggregator.cuda()
    
    print_cz("===> Loading datasets", f=log_file)
    """ dir and csv provided in functions"""
    train_loader = dataloader.get_dataloader(
        stage='train', 
        test_split=args.test_split,
        wsi_shuffle=True)
    test_loader = dataloader.get_dataloader(
        stage='test', 
        test_split=args.test_split,
        wsi_shuffle=False)

    print_cz("===> Setting Optimizer", f=log_file)
    if args.optim in ['Adam', 'adam']:
        optimizer_extractor_fam = torch.optim.Adam(params=extractor_fam.parameters(), 
            lr=args.lr * args.lr_fam_factor, 
            weight_decay=args.weight_decay)
        optimizer_extractor_tw = torch.optim.Adam(params=extractor_tw.parameters(), 
            lr=args.lr * args.lr_extractor_factor, 
            weight_decay=args.weight_decay)
        optimizer_extractor_mid = torch.optim.Adam(params=extractor_mid.parameters(), 
            lr=args.lr * args.lr_extractor_factor, 
            weight_decay=args.weight_decay)
        optimizer_extractor_bottom = torch.optim.Adam(params=extractor_bottom.parameters(), 
            lr=args.lr * args.lr_extractor_factor, 
            weight_decay=args.weight_decay)
        optimizer_aggregator = torch.optim.Adam(params=aggregator.parameters(), 
            lr=args.lr * args.lr_agg_factor, 
            weight_decay=args.weight_decay)
    elif args.optim in ['SGD', 'sgd']:
        optimizer_extractor_fam = torch.optim.SGD(params=extractor_fam.parameters(), 
            lr=args.lr * args.lr_fam_factor, 
            weight_decay=args.weight_decay, 
            momentum=args.momen)
        optimizer_extractor_tw = torch.optim.SGD(params=extractor_tw.parameters(), 
            lr=args.lr * args.lr_extractor_factor, 
            weight_decay=args.weight_decay, 
            momentum=args.momen)
        optimizer_extractor_mid = torch.optim.SGD(params=extractor_mid.parameters(), 
            lr=args.lr * args.lr_extractor_factor, 
            weight_decay=args.weight_decay, 
            momentum=args.momen)
        optimizer_extractor_bottom = torch.optim.SGD(params=extractor_bottom.parameters(), 
            lr=args.lr * args.lr_extractor_factor, 
            weight_decay=args.weight_decay, 
            momentum=args.momen)
        optimizer_aggregator = torch.optim.SGD(params=aggregator.parameters(), 
            lr=args.lr * args.lr_agg_factor, 
            weight_decay=args.weight_decay, 
            momentum=args.momen)

    print_cz("===> Training", f=log_file)
    train(train_loader=train_loader, 
        test_loader=test_loader, 
        extractor_fam=extractor_fam,
        extractor_tw=extractor_tw,
        extractor_mid=extractor_mid,
        extractor_bottom=extractor_bottom,
        aggregator=aggregator,
        criterion=criterion, 
        criterion_sparse=criterion_sparse,
        criterion_semantic=criterion_semantic,
        optimizer_extractor_fam=optimizer_extractor_fam, 
        optimizer_extractor_tw=optimizer_extractor_tw, 
        optimizer_extractor_mid=optimizer_extractor_mid, 
        optimizer_extractor_bottom=optimizer_extractor_bottom, 
        optimizer_aggregator=optimizer_aggregator,
        lr=args.lr, 
        lr_fam_factor=args.lr_fam_factor, 
        lr_extractor_factor=args.lr_extractor_factor, 
        lr_agg_factor=args.lr_agg_factor, 
        epochs=args.epochs, 
        soft_batch_size=args.soft_batch_size, 
        lr_step=args.lr_step, 
        lr_gamma=args.lr_gamma, 
        patch_limits=args.fam_limits,
        logfile=log_file, 
        save_path=log_dir+'/',
        )

    print_cz(str(time.time()-t), f=log_file)
    log_file.close()

