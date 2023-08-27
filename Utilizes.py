
import math
import numpy
import lpips
import torch
import time
import fast_soft_sort.pytorch_ops as ops
import scipy.io as scio

def DataShuffleBox(org_num=200, shuffle_rate=0.8, fix=False):

    seq = numpy.arange(org_num)
    if not fix:
        numpy.random.shuffle(seq)
    # else:
    #     seq = numpy.arange(org_num-1, -1, -1)
    seq_s = seq.tolist()
    train_num = numpy.round(org_num * (1-shuffle_rate))
    # train_num = numpy.floor(org_num * shuffle_rate)
    seq_eval = seq_s[0:int(train_num)]
    seq_train = seq_s[int(train_num):]

    # train_num = numpy.round(org_num * shuffle_rate)
    # D = scio.loadmat('/home/vista/Documents/ZhouZehong/VISOR_plus/version3/data/LIVECinfo.mat')
    # seq_s = D['index'][0][:]
    # seq_train = seq_s[0:int(train_num)]
    # seq_eval = seq_s[int(train_num):]

    return seq_eval, seq_train


def PerceptualLoss(img1, img2, cuda_flg=False):

    loss_fn = lpips.LPIPS(net='alex', spatial=True, verbose=False)
    if cuda_flg:
        loss_fn = loss_fn.cuda()

    loss_fn.requires_grad = False
    # with torch.no_grad():
    loss = loss_fn(img1, img2)
    # loss = loss_fn.forward(img1, img2)

    return loss.mean()


def cal_time(func):

    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print("%s running time: %s mins." % (func.__name__, (t2-t1) / 60))
        return result

    return wrapper

def PLCC_Rank_Loss(data1, data2):

    dat1 = data1.squeeze(0)
    dat2 = data2.squeeze(0)
    m1 = torch.mean(dat1)
    m2 = torch.mean(dat2)
    plcc = (torch.dot((dat1 - m1), (dat2 - m2)) + 1e-6) / (torch.sqrt(torch.dot((dat1 - m1), (dat1 - m1))
                                                                * torch.dot((dat2 - m2), (dat2 - m2))) + 1e-6)

    return (1-plcc)/2


def SRCC_Rank_Loss(dat1, dat2, **kw):
    # dat1 = data1.squeeze(0)
    # dat2 = data2.squeeze(0)
    dat1 = dat1.unsqueeze(-1).cpu()
    dat2 = dat2.unsqueeze(-1).cpu()
    assert dat1.size(0) > 1
    dat1 = torch.t(dat1)
    dat1 = ops.soft_rank(dat1, **kw)
    dat1 = dat1 - dat1.mean()
    dat1 = dat1 / dat1.norm()
    dat2 = torch.t(dat2)
    dat2 = ops.soft_rank(dat2, **kw)
    dat2 = dat2 - dat2.mean()
    dat2 = dat2 / dat2.norm()
    dat1 = dat1.cuda()
    dat2 = dat2.cuda()
    return 1-(dat1 * dat2).sum()


def l1_regularization(model, l1_alpha):
    l1_loss = 0
    for name, paras in model.named_parameters():
        l1_loss += torch.sum(torch.abs(paras))
    return l1_alpha * l1_loss


def image_crop_and_stack(img, crop_size=[224,224], gap=224):

    img = img.squeeze(0)
    H = img.size(1)
    W = img.size(2)
    xnum = (H-crop_size[0]) // gap + 1
    ynum = (W-crop_size[1]) // gap + 1
    N = xnum * ynum
    for i in range(xnum):
        for j in range(ynum):
            x_start = i*gap
            y_start = j*gap
            patch = img[:, x_start:x_start+crop_size[0], y_start:y_start+crop_size[1]]
            if i+j == 0:
                patches = patch.unsqueeze(0)
            else:
                patches = torch.cat([patches, patch.unsqueeze(0)], dim=0)
    assert patches.size(0) == N
    return patches


def multi_image_crop_and_stack(imgs, crop_size=[224,224], gap=224):
    first_flg = True
    BatchNum = imgs.size(0)
    Nbox = []
    for bn in range(BatchNum):
        img = imgs[bn]
        H = img.size(1)
        W = img.size(2)
        xnum = (H-crop_size[0]) // gap + 1
        ynum = (W-crop_size[1]) // gap + 1
        N = xnum * ynum
        for i in range(xnum):
            for j in range(ynum):
                x_start = i*gap
                y_start = j*gap
                patch = img[:, x_start:x_start+crop_size[0], y_start:y_start+crop_size[1]]
                if first_flg:
                    patches = patch.unsqueeze(0)
                    first_flg = False
                else:
                    patches = torch.cat([patches, patch.unsqueeze(0)], dim=0)
        assert patches.size(0) == N
        Nbox.append(N)
    for idx in range(len(Nbox)):
        if idx>0:
            Nbox[idx] = Nbox[idx] + Nbox[idx-1]
    return patches, Nbox


def multi_mean(scores, Nbox):
    slist = []
    for i in range(len(Nbox)):
        if i == 0:
            start = 0
        else:
            start = Nbox[i-1]
        stmp = torch.mean(scores[start:Nbox[i]])
        slist.append(stmp.item())
    return slist

