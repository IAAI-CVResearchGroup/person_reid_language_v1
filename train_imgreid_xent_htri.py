from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import scipy.io as sio
import pdb
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from torchreid import data_manager

from torchreid import transforms as T
from torchreid import models
from torchreid.losses import CrossEntropyLabelSmooth, TripletLoss
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.logger import Logger
from torchreid.utils.torchtools import count_num_param
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.samplers import RandomIdentitySampler
from torchreid.optimizers import init_optim

from eval.re_ranking_feature import re_ranking
from eval.eval_AlignedReID import eval_map_cmc


from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')
# Datasets
parser.add_argument('--root', type=str, default='data',
                    help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0,
                    help="split index")
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
# Results
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--print-freq', type=int, default=10,
                    help="print frequency")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Training options
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[20, 40], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
# loss
parser.add_argument('--loss', type=str, default='xent_only',
                    choices=['xent_only','htri_only','xent_htri'])
parser.add_argument('--lambda-xent', type=float, default=1,
                    help="weight to balance cross entropy loss")
parser.add_argument('--lambda-htri', type=float, default=1,
                    help="weight to balance hard triplet loss")
parser.add_argument('--label-smooth', action='store_true',
                    help="use label smoothing regularizer in cross entropy loss")
parser.add_argument('--margin', type=float, default=0.3,
                    help="margin for triplet loss")
# Validating/Testing options
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--test-batch', default=100, type=int,
                    help="test batch size")
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--rerank', action='store_true', help="perform re-ranking")
parser.add_argument('--vis-ranked-res', action='store_true',
                    help="visualize ranked results, only available in evaluation mode (default: False)")
# Others
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')


args = parser.parse_args()
# TODO: Train
if True:
    args = parser.parse_args(['--root', '/data/jun.xu/',
                              '--dataset','cuhk03',#'market1501',#'cuhk03',
                              '--cuhk03-labeled',
                              '--cuhk03-classic-split',
                              #=================
                              '--save-dir', '/data/jun.xu/log/log_cuhkL_v1_2',
                              #=================
                              '--arch','resnet50',
                              #=================
                              '--gpu-devices','1',
                              #'--load-weights', '/data/jun.xu/log/log_cuhkL_v1_2/checkpoint_ep110.pth.tar',
                              '--max-epoch', '500',
                              '--start-epoch', '0',
                              '--lr','0.0001',
                              '--stepsize','20',
                              '--gamma','0.1',
                              '--loss','xent_htri',#'xent_only',#'xent_htri',
                              '--lambda-xent','1',# for xent loss
                              '--margin','0.3',# for htri loss
                              '--lambda-htri','1',# for htri loss
                              #=================
                              '--start-eval', '0',
                              '--eval-step', '5',
                              ])

# TODO: Test
if False:
    args = parser.parse_args(['--evaluate',
                              '--root', '/data/jun.xu/',
                              '--dataset', 'cuhk03',
                              '--cuhk03-labeled',
                              '--cuhk03-classic-split'
                              '--load-weights', '/data/jun.xu/log/checkpoint_ep150.pth.tar',
                              '--gpu-devices','1',
                              '--rerank'
                              ])



if args.dataset == 'cuhk03':
    from torchreid.dataset_loader import ImageDataset #for cuhk
elif args.dataset == 'market1501':
    from torchreid.dataset_loader_market import ImageDataset # for market1501
elif args.dataset == 'dukemtmcreid':
    from torchreid.dataset_loader_duke import ImageDataset #for duke

if args.dataset == 'cuhk03':
    separate_camera_set=True
    single_gallery_shot=True
    first_match_break=False


# Parameter for reranking
reranking_lambda = 0.3
reranking_k1 = 20
reranking_k2 = 6

# Path for saving features
mat_path = 'save_mat/'
if os.path.exists(mat_path) == False:
    os.mkdir(mat_path)


writer = SummaryWriter(args.save_dir + '/')


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False


    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")


    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_imgreid_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )


    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, args.train_batch, args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    # Loss functions
    if args.label_smooth:
        criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    else:
        criterion_xent = nn.CrossEntropyLoss()
    criterion_htri = TripletLoss(margin=args.margin)


    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

    if args.load_weights:
        # load pretrained weights but ignore layers that don't match in size
        if check_isfile(args.load_weights):
            checkpoint = torch.load(args.load_weights)
            pretrain_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume:
        if check_isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            rank1 = checkpoint['rank1']
            print("Loaded checkpoint from '{}'".format(args.resume))
            print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, rank1))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)
        if args.vis_ranked_res:
            visualize_ranked_results(
                distmat, dataset,
                save_dir=osp.join(args.save_dir, 'ranked_results'),
                topk=20,
            )
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")


    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:

            rank1 = 0
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
               'state_dict': state_dict,
               'rank1': rank1,
               'epoch': epoch,
            }, False, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))
            print("model saved")

            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu, epoch=epoch)

            writer.add_scalars('train_val_top1', {'rank1': rank1}, epoch)

            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

    writer.export_scalars_to_json(args.save_dir + "/tensor_train.json")
    writer.close()


def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_xent = AverageMeter()
    losses_htri = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids, _, caps, caps_raw) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids, caps = imgs.cuda(), pids.cuda(), caps.cuda()

        outputs, features = model(imgs, caps)

        # Loss
        if args.loss == 'xent_only':
            xent_loss = criterion_xent(outputs, pids)
            loss = xent_loss
        elif args.loss == 'htri_only':
            htri_loss = criterion_htri(features, pids)
            loss = htri_loss
        elif args.loss == 'xent_htri':
            xent_loss = criterion_xent(outputs, pids)
            htri_loss = criterion_htri(features, pids)
            loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss

            losses_xent.update(xent_loss.item(), pids.size(0))
            losses_htri.update(htri_loss.item(), pids.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), pids.size(0))


        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'xent_loss={xent_loss.avg:.4f}, htri_loss={htri_loss.avg:.4f}, '
                  .format(
                epoch + 1, batch_idx + 1, len(trainloader),
                batch_time=batch_time,
                data_time=data_time, loss=losses, xent_loss=losses_xent, htri_loss=losses_htri))


        end = time.time()

    writer.add_scalars('train_average_loss', {'loss': losses.avg}, epoch)


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False, epoch=0):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, caps, caps_raw) in enumerate(queryloader):
            if use_gpu:
                imgs, caps = imgs.cuda(), caps.cuda()

            end = time.time()

            features = model(imgs, caps)

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, caps, _) in enumerate(galleryloader):
            if use_gpu:
                imgs, caps = imgs.cuda(), caps.cuda()

            end = time.time()

            features = model(imgs, caps)

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))


    if args.rerank:
        print('re-ranking (Euclidean distance)')
        distmat = re_ranking(qf, gf, k1=reranking_k1, k2=reranking_k2, lambda_value=reranking_lambda)

    else:
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()

    if args.evaluate:
        sio.savemat(mat_path + 'dismat.mat', {'dismat':distmat})
        sio.savemat(mat_path + 'g_pids.mat', {'g_pids':g_pids})
        sio.savemat(mat_path + 'q_pids.mat', {'q_pids':q_pids})
        sio.savemat(mat_path + 'g_camids.mat', {'g_camids':g_camids})
        sio.savemat(mat_path + 'q_camids.mat', {'q_camids':q_camids})


    print("Computing CMC and mAP")
    if args.dataset == 'market1501':
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)
    elif args.dataset == 'cuhk03':
        mAP, cmc = eval_map_cmc(
            distmat,
            q_ids=q_pids, g_ids=g_pids,
            q_cams=q_camids, g_cams=g_camids,
            separate_camera_set=separate_camera_set,
            single_gallery_shot=single_gallery_shot,
            first_match_break=first_match_break,
            topk=20)
    elif args.dataset == 'dukemtmcreid':
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)


    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print(cmc)
    print("------------------")


    if return_distmat:
        return distmat

    return cmc[0]


if __name__ == '__main__':
    main()
