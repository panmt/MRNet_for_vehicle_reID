from __future__ import print_function
from __future__ import division

import os
import sys
import os.path as osp
import warnings
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from tool.data_manager import ImageDataManager
from model.big_model import Model
from tool.losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from tool.utils.iotools import check_isfile
from tool.utils.avgmeter import AverageMeter
from tool.utils.loggers import Logger, RankLogger
from tool.utils.torchtools import count_num_param, accuracy, \
    load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from tool.utils.visualtools import visualize_ranked_results
from tool.utils.generaltools import set_random_seed
from tool.eval_metrics import evaluate
from tool.optimizers import init_optimizer
from tool.lr_schedulers import init_lr_scheduler, GCN_lr_scheduler
from PIL import Image
import torchvision.models as models
from torchvision import transforms
from skimage import io
#from tool import transforms_image
from skimage import transform
parser = argument_parser()
args = parser.parse_args()


def main():
    global args

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()  # ~~~testloader_dict have query and gallery

    print('Initializing model: {}'.format(args.arch))
    model = Model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent', 'htri'},
                  pretrained=not args.no_pretrained, use_gpu=use_gpu)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model = model.cuda() if use_gpu else model
    kp_ckpt = './model/keypoint_model/best_fine_kp_checkpoint.pth.tar'  ## pre-trained kp_model
    model.kp_net.load_state_dict(torch.load(kp_ckpt)['net_state_dict'])
    print('kp_net Checkpoint :{} is Loaded!'.format(kp_ckpt))
    for param in model.kp_net.parameters():
        param.requires_grad = False
    print('kp_net have been initialized with pre-trained weights and are frozen!')

    criterion_xent = CrossEntropyLoss(num_classes=dm.num_train_pids, label_smooth=args.label_smooth)
    criterion_htri = TripletLoss(margin=args.margin)
    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

    if args.evaluate:
        print('Evaluate only')

        for name in args.target_names:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, 'ranked_results', name),
                    topk=20
                )
        return

    time_start = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    print('=> Start training')
    '''
    if args.fixbase_epoch > 0:
        print('Train {} for {} epochs while keeping other layers frozen'.format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, fixbase=True)

        print('Done. All layers are open to train for {} epochs'.format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)
    '''
    best = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch:
            print('=> Test')

            for name in args.target_names:
                print('Evaluating {} ...'.format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                rank1 = test(model, queryloader, galleryloader, use_gpu)
                ranklogger.write(name, epoch + 1, rank1)
            if rank1 > best:
                best = rank1
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'rank1': rank1,
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'optimizer': optimizer.state_dict(),
                }, args.save_dir)

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))
    ranklogger.show_summary()

def generate_box(fine_kp, orientation):
    """
    Randomly visualize the estimated key-points and their respective ground-truth maps
    :param outputs: the estimated key-points
    :param maps: the ground-truth maps
    :param inputs: the tensor containing the normalized image data
    :return: visualize the heatmaps
    """
    ### fine_kp: B*21*56*56  orientation: B*8
    fine_kp = fine_kp.cpu().numpy()
    N, C, _, _ = fine_kp.shape
    kp = fine_kp[:, :20, :, :]
    _, orientation = torch.max(orientation, 1)
    orientation = orientation.cpu().numpy()
    patchs = np.zeros((N, 7, 4))
    for n in range(N):
        t = np.empty(shape=[0, 2])
        c = np.zeros((1, 2))
        m = kp[n].sum()
        keypoint = kp[n]
        predicted_orientation = orientation[n]
        for k in range(20):
            keypoint[k] = keypoint[k] / m
            # print(map_out[i])
            b = np.where(keypoint[k] == np.max(keypoint[k]))
            c[0][0] = b[1][0]
            c[0][1] = b[0][0]
            t = np.append(t, c, axis=0)

        # print(t)
        seven = np.zeros((7, 4))
        if predicted_orientation == 0:
            seven[0][0] = min(t[15][1], t[14][1], t[13][1], t[12][1])
            seven[0][1] = max(t[15][1], t[14][1], t[13][1], t[12][1])
            seven[0][2] = min(t[15][0], t[14][0], t[13][0], t[12][0])
            seven[0][3] = max(t[15][0], t[14][0], t[13][0], t[12][0])
            seven[1][0] = min(t[12][1], t[13][1], t[6][1], t[7][1])
            seven[1][1] = max(t[12][1], t[13][1], t[6][1], t[7][1])
            seven[1][2] = min(t[12][0], t[13][0], t[6][0], t[7][0])
            seven[1][3] = max(t[12][0], t[13][0], t[6][0], t[7][0])
            seven[2][0] = min(t[6][1], t[7][1], t[9][1], t[5][1], t[4][1])
            seven[2][1] = max(t[6][1], t[7][1], t[9][1], t[5][1], t[4][1])
            seven[2][2] = min(t[6][0], t[7][0], t[9][0], t[5][0], t[4][0])
            seven[2][3] = max(t[6][0], t[7][0], t[9][0], t[5][0], t[4][0])
            seven[3:, :] = 0
            # print("0")
        elif predicted_orientation == 1:
            seven[0][0] = min(t[15][1], t[14][1], t[13][1], t[12][1])
            seven[0][1] = max(t[15][1], t[14][1], t[13][1], t[12][1])
            seven[0][2] = min(t[15][0], t[14][0], t[13][0], t[12][0])
            seven[0][3] = max(t[15][0], t[14][0], t[13][0], t[12][0])
            seven[1:5, :] = 0
            seven[5][0] = min(t[14][1], t[15][1], t[16][1], t[17][1])
            seven[5][1] = max(t[14][1], t[15][1], t[16][1], t[17][1])
            seven[5][2] = min(t[14][0], t[15][0], t[16][0], t[17][0])
            seven[5][3] = max(t[14][0], t[15][0], t[16][0], t[17][0])
            seven[6][0] = min(t[19][1], t[16][1], t[17][1])
            seven[6][1] = max(t[19][1], t[16][1], t[17][1])
            seven[6][2] = min(t[19][0], t[16][0], t[17][0])
            seven[6][3] = max(t[19][0], t[16][0], t[17][0])
            # print("1")
        elif predicted_orientation == 2:
            seven[0:3, :] = 0
            seven[3][0] = min(t[0][1], t[1][1], t[13][1], t[14][1])
            seven[3][1] = max(t[0][1], t[1][1], t[13][1], t[14][1])
            seven[3][2] = min(t[0][0], t[1][0], t[13][0], t[14][0])
            seven[3][3] = max(t[0][0], t[1][0], t[13][0], t[14][0])
            seven[4][0] = min(t[0][1], t[1][1], t[13][1], t[14][1])
            seven[4][1] = max(t[0][1], t[1][1], t[13][1], t[14][1])
            seven[4][2] = min(t[0][0], t[1][0], t[13][0], t[14][0])
            seven[4][3] = max(t[0][0], t[1][0], t[13][0], t[14][0])
            seven[5:, :] = 0
            # print("2")
        elif predicted_orientation == 3:
            seven[0][0] = min(t[15][1], t[14][1], t[13][1], t[12][1])
            seven[0][1] = max(t[15][1], t[14][1], t[13][1], t[12][1])
            seven[0][2] = min(t[15][0], t[14][0], t[13][0], t[12][0])
            seven[0][3] = max(t[15][0], t[14][0], t[13][0], t[12][0])
            seven[1][0] = min(t[12][1], t[13][1], t[6][1], t[7][1])
            seven[1][1] = max(t[12][1], t[13][1], t[6][1], t[7][1])
            seven[1][2] = min(t[12][0], t[13][0], t[6][0], t[7][0])
            seven[1][3] = max(t[12][0], t[13][0], t[6][0], t[7][0])
            seven[2][0] = min(t[6][1], t[7][1], t[9][1], t[5][1], t[4][1])
            seven[2][1] = max(t[6][1], t[7][1], t[9][1], t[5][1], t[4][1])
            seven[2][2] = min(t[6][0], t[7][0], t[9][0], t[5][0], t[4][0])
            seven[2][3] = max(t[6][0], t[7][0], t[9][0], t[5][0], t[4][0])
            seven[3][0] = min(t[0][1], t[1][1], t[13][1], t[14][1])
            seven[3][1] = max(t[0][1], t[1][1], t[13][1], t[14][1])
            seven[3][2] = min(t[0][0], t[1][0], t[13][0], t[14][0])
            seven[3][3] = max(t[0][0], t[1][0], t[13][0], t[14][0])
            seven[4][0] = min(t[0][1], t[1][1], t[13][1], t[14][1])
            seven[4][1] = max(t[0][1], t[1][1], t[13][1], t[14][1])
            seven[4][2] = min(t[0][0], t[1][0], t[13][0], t[14][0])
            seven[4][3] = max(t[0][0], t[1][0], t[13][0], t[14][0])
            seven[5:, :] = 0
            # print("3")
        elif predicted_orientation == 4:
            seven[0][0] = min(t[15][1], t[14][1], t[13][1], t[12][1])
            seven[0][1] = max(t[15][1], t[14][1], t[13][1], t[12][1])
            seven[0][2] = min(t[15][0], t[14][0], t[13][0], t[12][0])
            seven[0][3] = max(t[15][0], t[14][0], t[13][0], t[12][0])
            seven[1:3, :] = 0
            seven[3][0] = min(t[0][1], t[1][1], t[13][1], t[14][1])
            seven[3][1] = max(t[0][1], t[1][1], t[13][1], t[14][1])
            seven[3][2] = min(t[0][0], t[1][0], t[13][0], t[14][0])
            seven[3][3] = max(t[0][0], t[1][0], t[13][0], t[14][0])
            seven[4][0] = min(t[0][1], t[1][1], t[13][1], t[14][1])
            seven[4][1] = max(t[0][1], t[1][1], t[13][1], t[14][1])
            seven[4][2] = min(t[0][0], t[1][0], t[13][0], t[14][0])
            seven[4][3] = max(t[0][0], t[1][0], t[13][0], t[14][0])
            seven[5][0] = min(t[14][1], t[15][1], t[16][1], t[17][1])
            seven[5][1] = max(t[14][1], t[15][1], t[16][1], t[17][1])
            seven[5][2] = min(t[14][0], t[15][0], t[16][0], t[17][0])
            seven[5][3] = max(t[14][0], t[15][0], t[16][0], t[17][0])
            seven[6][0] = min(t[19][1], t[16][1], t[17][1])
            seven[6][1] = max(t[19][1], t[16][1], t[17][1])
            seven[6][2] = min(t[19][0], t[16][0], t[17][0])
            seven[6][3] = max(t[19][0], t[16][0], t[17][0])
            # print("4")
        elif predicted_orientation == 5:
            seven[0:3, :] = 0
            seven[3][0] = min(t[12][1], t[15][1], t[2][1], t[3][1])
            seven[3][1] = max(t[12][1], t[15][1], t[2][1], t[3][1])
            seven[3][2] = min(t[12][0], t[15][0], t[2][0], t[3][0])
            seven[3][3] = max(t[12][0], t[15][0], t[2][0], t[3][0])
            seven[4][0] = min(t[12][1], t[15][1], t[2][1], t[3][1])
            seven[4][1] = max(t[12][1], t[15][1], t[2][1], t[3][1])
            seven[4][2] = min(t[12][0], t[15][0], t[2][0], t[3][0])
            seven[4][3] = max(t[12][0], t[15][0], t[2][0], t[3][0])
            seven[5:, :] = 0
            # print("5")
        elif predicted_orientation == 6:
            seven[0][0] = min(t[15][1], t[14][1], t[13][1], t[12][1])
            seven[0][1] = max(t[15][1], t[14][1], t[13][1], t[12][1])
            seven[0][2] = min(t[15][0], t[14][0], t[13][0], t[12][0])
            seven[0][3] = max(t[15][0], t[14][0], t[13][0], t[12][0])
            seven[1][0] = min(t[12][1], t[13][1], t[6][1], t[7][1])
            seven[1][1] = max(t[12][1], t[13][1], t[6][1], t[7][1])
            seven[1][2] = min(t[12][0], t[13][0], t[6][0], t[7][0])
            seven[1][3] = max(t[12][0], t[13][0], t[6][0], t[7][0])
            seven[2][0] = min(t[6][1], t[7][1], t[9][1], t[5][1], t[4][1])
            seven[2][1] = max(t[6][1], t[7][1], t[9][1], t[5][1], t[4][1])
            seven[2][2] = min(t[6][0], t[7][0], t[9][0], t[5][0], t[4][0])
            seven[2][3] = max(t[6][0], t[7][0], t[9][0], t[5][0], t[4][0])
            seven[3][0] = min(t[12][1], t[15][1], t[2][1], t[3][1])
            seven[3][1] = max(t[12][1], t[15][1], t[2][1], t[3][1])
            seven[3][2] = min(t[12][0], t[15][0], t[2][0], t[3][0])
            seven[3][3] = max(t[12][0], t[15][0], t[2][0], t[3][0])
            seven[4][0] = min(t[12][1], t[15][1], t[2][1], t[3][1])
            seven[4][1] = max(t[12][1], t[15][1], t[2][1], t[3][1])
            seven[4][2] = min(t[12][0], t[15][0], t[2][0], t[3][0])
            seven[4][3] = max(t[12][0], t[15][0], t[2][0], t[3][0])
            seven[5:, :] = 0
            # print("6")
        else:
            seven[0][0] = min(t[15][1], t[14][1], t[13][1], t[12][1])
            seven[0][1] = max(t[15][1], t[14][1], t[13][1], t[12][1])
            seven[0][2] = min(t[15][0], t[14][0], t[13][0], t[12][0])
            seven[0][3] = max(t[15][0], t[14][0], t[13][0], t[12][0])
            seven[1:3, :] = 0
            seven[3][0] = min(t[12][1], t[15][1], t[2][1], t[3][1])
            seven[3][1] = max(t[12][1], t[15][1], t[2][1], t[3][1])
            seven[3][2] = min(t[12][0], t[15][0], t[2][0], t[3][0])
            seven[3][3] = max(t[12][0], t[15][0], t[2][0], t[3][0])
            seven[4][0] = min(t[12][1], t[15][1], t[2][1], t[3][1])
            seven[4][1] = max(t[12][1], t[15][1], t[2][1], t[3][1])
            seven[4][2] = min(t[12][0], t[15][0], t[2][0], t[3][0])
            seven[4][3] = max(t[12][0], t[15][0], t[2][0], t[3][0])
            seven[5][0] = min(t[14][1], t[15][1], t[16][1], t[17][1])
            seven[5][1] = max(t[14][1], t[15][1], t[16][1], t[17][1])
            seven[5][2] = min(t[14][0], t[15][0], t[16][0], t[17][0])
            seven[5][3] = max(t[14][0], t[15][0], t[16][0], t[17][0])
            seven[6][0] = min(t[19][1], t[16][1], t[17][1])
            seven[6][1] = max(t[19][1], t[16][1], t[17][1])
            seven[6][2] = min(t[19][0], t[16][0], t[17][0])
            seven[6][3] = max(t[19][0], t[16][0], t[17][0])
            # print("7")

        patchs[n] = seven
    return patchs, orientation

def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    LSTM_losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # for p in model.parameters():
    #     p.requires_grad = True    # open all layers
    end = time.time()
    for batch_idx, (image_224, image_56, pids, camids, img_paths) in enumerate(trainloader):
        data_time.update(time.time() - end)
        if use_gpu:
            image_224, image_56, pids = image_224.cuda(), image_56.cuda(), pids.cuda()

        coarse_kp, fine_kp, orientation = model.kp_net(image_224, image_56)
        patchs, orientation = generate_box(fine_kp, orientation)
        model.train()
        outputs, features, feature_LSTM = model(image_224, patchs)
        if isinstance(outputs, (tuple, list)):
            xent_loss = DeepSupervision(criterion_xent, outputs, pids)
        else:
            xent_loss = criterion_xent(outputs, pids)

        if isinstance(features, (tuple, list)):
            htri_loss = DeepSupervision(criterion_htri, features, pids)
        else:
            htri_loss = criterion_htri(features, pids)

        LSTM_loss = criterion_htri(feature_LSTM, pids)
        loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss + args.lambda_LSTM * LSTM_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        xent_losses.update(xent_loss.item(), pids.size(0))
        htri_losses.update(htri_loss.item(), pids.size(0))
        LSTM_losses.update(LSTM_loss.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0])

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Xent {xent.val:.4f} ({xent.avg:.4f})\t'
                  'Htri {htri.val:.4f} ({htri.avg:.4f})\t'
                  'LSTMloss {LSTMloss.val:.4f} ({LSTMloss.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader),
                batch_time=batch_time,
                data_time=data_time,
                xent=xent_losses,
                htri=htri_losses,
                LSTMloss=LSTM_losses,
                acc=accs
            ))

        end = time.time()


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (image_224, image_56, pids, camids, img_paths) in enumerate(queryloader):
            if use_gpu:
                image_224, image_56 = image_224.cuda(), image_56.cuda()
            end = time.time()
            coarse_kp, fine_kp, orientation = model.kp_net(image_224, image_56)
            patchs, orientation = generate_box(fine_kp, orientation)
            feat = model(image_224, patchs)  # B*2048
            batch_time.update(time.time() - end)

            features = feat.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (image_224, image_56, pids, camids, img_paths) in enumerate(galleryloader):
            if use_gpu:
                image_224, image_56 = image_224.cuda(), image_56.cuda()
            end = time.time()
            coarse_kp, fine_kp, orientation = model.kp_net(image_224, image_56)
            patchs, orientation = generate_box(fine_kp, orientation)
            feat = model(image_224, patchs)  # B*2048
            batch_time.update(time.time() - end)

            features = feat.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print('Computing CMC and mAP')
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == '__main__':
    main()
