# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:33:18 2020

@author: marsh26macro
"""

import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Net_Big import *
from Net_New import VISORNetV3
from DATA_READ import ReadIQAFolder
import torchvision.transforms as transforms
import ResultEvaluate as RE
from Utilizes import *
import numpy as np
import logging
from scipy import stats
import torch.optim.lr_scheduler as lr_scheduler
from thop import profile

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Regression")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Default=1e-4")
parser.add_argument("--lrratio", type=float, default=10, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument("--resume", default='', type=str,
                    help="path to latest checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="number of threads for data loader to use")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--database", default='livec', type=str, help="momentum")
parser.add_argument("--gpuid", default='0', type=str, help="id of GPU")
parser.add_argument("--logname", default='tmp', type=str, help="name of log file")
parser.add_argument("--hpinfo", action="store_false", help="put the params in the log or not")
parser.add_argument("--rcrop", action="store_false", help="random crop in training set")
parser.add_argument("--aug", action="store_false", help="augmentation in training set")
parser.add_argument("--nexp", default=0, type=int, help="number of the experiment")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")
parser.add_argument("--w-srcc", "--ws", default=1, type=float, help="weight of SRCC Loss")
parser.add_argument("--w-plcc", "--wp", default=1, type=float, help="weight of PLCC Loss")
parser.add_argument("--w-mse", "--wm", default=1, type=float, help="weight of MSE Loss")


@cal_time
def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuid
    logging.basicConfig(filename=opt.logname, filemode='a', level='INFO')
    info_line = 'To be Test: ' + opt.database
    logging.info(info_line)
    logging.info(opt)

    local_dir = '/media/vista/ZhouZehong13602684927/ImageDatasetForIQA/'
    spaq_dir = '/media/vista/0F010DD80F010DD8/data_by_zzh/IQAdataset/'
    # local_dir = '/mnt/hdd1/zzh/IQAdataset/'
    # spaq_dir = '/mnt/hdd1/zzh/IQAdataset/'

    matpath = {
        'live': ['./data/LIVEinfo.mat', local_dir + 'LIVE/databaserelease2', 100],
        'csiq': ['./data/CSIQinfo.mat', local_dir + 'CSIQ', 1],
        'tid2013': ['./data/TID2013info.mat', local_dir + 'TID2013', 10],
        'livec': ['./data/LIVECinfo_new.mat', local_dir + 'LIVEC/ChallengeDB_release', 100],
        'kadid10k': ['./data/KADID10Kinfo.mat', local_dir + 'KADID10k/kadid10k/kadid10k', 10],
        'koniq10k': ['./data/KonIQ10Kinfo.mat', local_dir + 'KonIQ10K', 10],
        'spaq': ['./data/SPAQinfo.mat', spaq_dir + 'SPAQ/SPAQ zip', 100],
        'flive': ['./data/LIVEFBinfo.mat', spaq_dir + 'LIVE-FB/FLIVE_Database-master/database', 100],
    }

    train_img_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    eval_img_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = 19980720
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    test_set = ReadIQAFolder(matpath=matpath[opt.database], nexp=opt.nexp, aug=False, random_crop=True,
                             img_transform=eval_img_tf, status='eval')
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    valid_set = ReadIQAFolder(matpath=matpath[opt.database], nexp=opt.nexp, aug=False, random_crop=True,
                             img_transform=eval_img_tf, status='valid')
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    print("===> Building model and Setting GPU")

    model = VISORNetV3(cout=256, dout=256).cuda()
    checkpoint = torch.load('./model/visor_premodel_epoch_28.pth')
    # model.CEnc.load_state_dict(checkpoint["enc_cont"].state_dict())
    model.DEnc.load_state_dict(checkpoint["enc_diff"].state_dict())

    criterion = nn.MSELoss()

    # Parameters Calculation
    # total1 = sum([param.nelement() for param in model.parameters()])
    # total2 = sum([param.nelement() for param in model0.parameters()])
    # total3 = sum([param.nelement() for param in modelc.parameters()])
    # print("Number of parameter: %.4fM" % ((total1+total2+total3) / 1e6))

    # FLOPS calculation
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, inputs=(input, ))
    # print(flops, params)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Training")
    best_Vres = [0, 0, 0, 0, 0]
    best_res = [0, 0, 0, 0, 0]

    for param in model.Reg.parameters():
        param.requires_grad = True
    for param in model.CEnc.parameters():
        param.requires_grad = False
    for param in model.DEnc.parameters():
        param.requires_grad = True

    # paras = [{'params': filter(lambda p: p.requires_grad, model.Reg.parameters()), 'lr': opt.lr * opt.lrratio},
    #          {'params': filter(lambda p: p.requires_grad, model.DEnc.parameters()), 'lr': opt.lr}
    #          ]
    # optimizer = optim.Adam(paras, weight_decay=opt.weight_decay)
    # optimizer = optim.Adam(paras, betas=(0.9, 0.999),
    #                        eps=1e-08, weight_decay=opt.weight_decay)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=opt.weight_decay)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=0.9)
    # scheduler = lr_scheduler(optimizer, 1e-4, 1e-3)

    train_set = ReadIQAFolder(matpath=matpath[opt.database], nexp=opt.nexp, aug=True, random_crop=True,
                              img_transform=train_img_tf, status='train')
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, pin_memory=True,
                              num_workers=4)

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):

        # Training
        train(train_loader, model, criterion, epoch, optimizer)
        # Update optimizer
        # lr = opt.lr / pow(10, (epoch // 6))
        # cur_lrratio = opt.lrratio
        # if epoch > 8:
        #     cur_lrratio = 1
        # paras = [{'params': filter(lambda p: p.requires_grad, model.Reg.parameters()), 'lr': lr * cur_lrratio},
        #          {'params': filter(lambda p: p.requires_grad, model.DEnc.parameters()), 'lr': opt.lr}
        #          ]
        # optimizer = torch.optim.Adam(paras, weight_decay=opt.weight_decay)

        # Testing
        if epoch % 1 == 0:
            # valid_res = test(valid_loader, model, criterion, epoch, status='valid')
            # eval_res = test(test_loader, model, criterion, epoch, status='test')
            valid_res = test_patches(valid_loader, model, criterion, epoch, status='valid')
            eval_res = test_patches(test_loader, model, criterion, epoch, status='test')
            if abs(valid_res[0]) > abs(best_Vres[0]):
                best_Vres = valid_res
                best_res = eval_res

        # if epoch % 1 == 0:
        #     save_checkpoint(model, epoch)

    print("===> BEST_Valid_Epoch[{}] SRCC: {:.4f} KRCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f}".format(best_Vres[4],
                                                                                           best_Vres[0], best_Vres[1],
                                                                                           best_Vres[2], best_Vres[3]))
    print("===> BEST_Test_Epoch[{}] SRCC: {:.4f} KRCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f}".format(best_res[4],
                                                                                           best_res[0], best_res[1],
                                                                                           best_res[2], best_res[3]))
    logging.info("===> BEST_Epoch[{}] SRCC: {:.4f} KRCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f}".format(best_res[4],
                                                                                                  best_res[0],
                                                                                                  best_res[1],
                                                                                                  best_res[2],
                                                                                                  best_res[3]))


def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    # lr = opt.lr * (0.1 ** (epoch // opt.step))
    if epoch % opt.step == 0:
        lr = opt.lr * 0.8
    else:
        lr = opt.lr

    return lr


@cal_time
def train(training_loader, model, criterion, epoch, optimizer):

    model.CEnc.eval()
    model.DEnc.train()
    model.Reg.train()

    lr = adjust_learning_rate(epoch)
    opt.lr = lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    avg_loss = 0
    g_loss1 = 0
    g_loss = 0
    pred_scores = []
    gt_scores = []

    for iteration, load_data in enumerate(training_loader, 1):
        img, org, mos, ids = load_data
        mos = mos.type(torch.FloatTensor)
        if opt.cuda:
            img = img.cuda()
            org = org.cuda()
            mos = mos.cuda()

        # NR
        score = model(img)
        # FR
        # score = model(img, org)

        pred_scores = pred_scores + score.cpu().tolist()
        gt_scores = gt_scores + mos.cpu().tolist()

        loss1 = criterion(score.squeeze(1), mos)
        loss2 = SRCC_Rank_Loss(score.squeeze(1), mos)
        loss3 = PLCC_Rank_Loss(score.squeeze(1), mos)
        lam1 = opt.w_mse
        lam2 = opt.w_srcc
        lam3 = opt.w_plcc
        # loss = (lam1 * loss1 + lam2 * loss2 + lam3 * loss3) / (lam1 + lam2 + lam3)
        loss = lam1 * loss1 + lam2 * loss2 + lam3 * loss3

        avg_loss += loss.detach()
        g_loss1 += loss1.detach()
        g_loss += loss.detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Scaler.scale(loss).backward()
        # Scaler.step(optimizer)
        # Scaler.update()

        gap = 100
        if iteration % gap == 0:
            print("===> Epoch[{}]({}/{}): LossA: {:.10f}".format(epoch, iteration, len(training_loader),
                                                                 avg_loss / gap))
            avg_loss = 0
    # scheduler.step()

    train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    print("===> Epoch[{}]: G_Loss: {:.10f} G_Loss1 {:.10f} TrainSRCC {:.4f}".format(epoch, g_loss / iteration,
                                                                   g_loss1 / iteration, train_srcc))



@cal_time
def test(test_loader, model, criterion, epoch, status='test'):

    model.CEnc.eval()
    model.DEnc.eval()
    model.Reg.eval()

    out_box = np.zeros(shape=(len(test_loader)))
    mos_box = np.zeros(shape=(len(test_loader)))

    tmp_out = []
    tmp_mos = []

    avg_loss = 0
    with torch.no_grad():
        for iteration, load_data in enumerate(test_loader, 1):
            img, org, mos, ids = load_data
            mos = mos.type(torch.FloatTensor)
            if opt.cuda:
                org = org.cuda()
                img = img.cuda()
                mos = mos.cuda()

            # score = model(img, org)
            score = model(img)
            loss = criterion(score.squeeze(1), mos)
            avg_loss += loss.item()

            tmp_out.append(score.cpu())
            tmp_mos.append(mos.cpu())

            if iteration % 1 == 0:
                out_box[iteration - 1] = np.mean(np.array(tmp_out))
                mos_box[iteration - 1] = np.mean(np.array(tmp_mos))
                tmp_out = []
                tmp_mos = []

    # print(out_box[:20])
    # print(mos_box[:20])

    srcc = RE.srocc(out_box, mos_box)
    # srcc, _ = spearmanr(out_box, mos_box)
    krcc = RE.kendallcc(out_box, mos_box)
    plcc = RE.pearsoncc(out_box, mos_box)
    # plcc, _ = pearsonr(out_box, mos_box)
    rmse = RE.rootMSE(out_box, mos_box)

    res = [srcc, krcc, plcc, rmse, epoch]

    if status == 'test':
        print("===> Epoch[{}] TEST Loss: {:.10f} SRCC: {:.4f} KRCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f}".format(epoch,
                                                                                                         (avg_loss / iteration),
                                                                                                         srcc, krcc,
                                                                                                         plcc, rmse))
    else:
        print("===> Epoch[{}] VALID Loss: {:.10f} SRCC: {:.4f} KRCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f}".format(epoch,
                                                                                                             (avg_loss / iteration),
                                                                                                             srcc, krcc,
                                                                                                             plcc, rmse))
    return res


@cal_time
def test_patches(test_loader, model, criterion, epoch, status='test'):
    # model.Backbone.eval()
    model.CEnc.eval()
    model.DEnc.eval()
    model.Reg.eval()

    avg_loss = 0
    pred_scores = []
    gt_scores = []
    with torch.no_grad():
        for iteration, load_data in enumerate(test_loader, 1):
            img, org, mos, ids = load_data
            mos = mos.type(torch.FloatTensor)
            if opt.cuda:
                img = img.cuda()
                mos = mos.cuda()
            score = model(img)
            loss = criterion(score.squeeze(1), mos)
            avg_loss += loss.item()

            pred_scores.append(float(score.item()))
            gt_scores = gt_scores + mos.cpu().tolist()

    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 25)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 25)), axis=1)
    srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    plcc, _ = stats.pearsonr(pred_scores, gt_scores)

    krcc = -1.0
    rmse = -1.0

    res = [srcc, krcc, plcc, rmse, epoch]

    if status == 'test':
        print("===> Epoch[{}] TEST Loss: {:.10f} SRCC: {:.4f} KRCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f}".format(epoch,
                                                                                                             (avg_loss / iteration),
                                                                                                             srcc, krcc,
                                                                                                             plcc, rmse))
    else:
        print("===> Epoch[{}] VALID Loss: {:.10f} SRCC: {:.4f} KRCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f}".format(epoch,
                                                                                                             (avg_loss / iteration),
                                                                                                             srcc, krcc,
                                                                                                             plcc, rmse))
    return res


@cal_time
def test_patches_old(test_loader, model, criterion, epoch, eval_res):
    # model.Backbone.eval()
    model.CEnc.eval()
    model.DEnc.eval()
    model.Reg.eval()

    out_box = []
    mos_box = []

    avg_loss = 0
    iteration_gap = opt.batchSize // 8
    img = None
    mos = None
    Nbox = []
    with torch.no_grad():
        for iteration, load_data in enumerate(test_loader, 1):
            simg, org, smos, ids = load_data
            smos = smos.type(torch.FloatTensor)
            if opt.cuda:
                simg = simg.cuda()
                smos = smos.cuda()

            patches = image_crop_and_stack(simg)
            Nbox.append(patches.size(0))

            if not isinstance(img, torch.Tensor):
                img = patches
                mos = smos
            else:
                img = torch.cat([img, patches], dim=0)
                mos = torch.cat([mos, smos], dim=0)

            if iteration % iteration_gap == 0 or iteration == len(test_loader):
                for idx in range(len(Nbox)):
                    if idx > 0:
                        Nbox[idx] = Nbox[idx] + Nbox[idx - 1]
                scores = model(img)
                score = multi_mean(scores, Nbox)
                out_box.extend(score)
                mos_box.extend(mos.cpu())

                loss = criterion(torch.Tensor(score), mos.cpu())
                avg_loss += loss

                img = None
                mos = None
                Nbox = []

    out_box = np.array(out_box)
    mos_box = np.array(mos_box)

    srcc = RE.srocc(out_box, mos_box)
    # srcc, _ = spearmanr(out_box, mos_box)
    krcc = RE.kendallcc(out_box, mos_box)
    plcc = RE.pearsoncc(out_box, mos_box)
    # plcc, _ = pearsonr(out_box, mos_box)
    rmse = RE.rootMSE(out_box, mos_box)

    if srcc > eval_res[0]:
        eval_res[0] = srcc
        eval_res[1] = krcc
        eval_res[2] = plcc
        eval_res[3] = rmse
        eval_res[4] = epoch
        # save_checkpoint(model0, model, epoch)

    print("===> Epoch[{}] TEST Loss: {:.10f} SRCC: {:.4f} KRCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f}".format(epoch,
                                                                                                         (avg_loss.item() / iteration),
                                                                                                         srcc, krcc,
                                                                                                         plcc, rmse))
    return eval_res


def save_checkpoint(model, epoch):
    model_folder = "checkpoint_finetune/"
    model_out_path = model_folder + "fmodel_best.pth"
    state = {"epoch": epoch, "enc_dist": model, "reg_net": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()











