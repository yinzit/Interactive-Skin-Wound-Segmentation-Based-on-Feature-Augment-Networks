from torch.utils.data import DataLoader
from wound import LinearLesion
import socket
from datetime import datetime
import random

import os
import cv2
from model.unet import UNet
from model.cpfnet import CPFNet
# from model.cenet import CE_Net
# from model.interactive_unet import CE_Net
from model.interactive import CE_Net
from model.fanet import FANet
from model.resnet34 import ResNet34
from model.att_unet import AttU_Net
from model.csnet import CSNet
from model.danet import DANet
from model.pspnet import PSPNet
# from model.deeplabv3 import DeepLabv3_plus
from model.deeplabV3_plus import DeepLab
from PIL import Image
import torch
import time
from tensorboardX import SummaryWriter
import tqdm
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import Image

import utils.utils as u
import utils.loss as LS
from utils.config import DefaultConfig
import torch.backends.cudnn as cudnn

# 验证
def val(args, model, dataloader):
    print('\n')
    print('Start Validation!')
    with torch.no_grad():  # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
        model.eval()
        tbar = tqdm.tqdm(dataloader, desc='\r')  # 进度条

        total_Dice = []
        total_Acc = []
        total_jaccard = []
        total_Sensitivity = []
        total_Specificity = []

        for i, (data, label) in enumerate(tbar):
        # for i, (data, label, edge, d, path) in enumerate(tbar):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()  # CPU和GPU的Tensor之间转换
                label = label.cuda()
                # d = d.cuda()

            # get RGB predict image
            # ---------------------------------------------- #
            # main_out = model(data)  # normal
            main_out, edge_out = model(data)
            # main_out = model(data, d)  # interactive
            # ---------------------------------------------- #

            Dice, Acc, jaccard, Sensitivity, Specificity=u.eval_single_seg(main_out, label)

            total_Dice += Dice
            total_Acc += Acc
            total_jaccard += jaccard
            total_Sensitivity += Sensitivity
            total_Specificity += Specificity

            dice = sum(total_Dice) / len(total_Dice)
            acc = sum(total_Acc) / len(total_Acc)
            jac = sum(total_jaccard) / len(total_jaccard)
            sen = sum(total_Sensitivity) / len(total_Sensitivity)
            spe = sum(total_Specificity) / len(total_Specificity)

            tbar.set_description(
                'Dice: %.3f, Acc: %.3f, Jac: %.3f, Sen: %.3f, Spe: %.3f' % (dice, acc, jac, sen, spe))

        print('Dice:', dice)
        print('Acc:', acc)
        print('Jac:', jac)
        print('Sen:', sen)
        print('Spe:', spe)
        return dice, acc, jac, sen, spe
    

def train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer):
    step = 0
    best_pred = 0.0
    # print(len((dataloader_train)))
    for epoch in range(args.num_epochs):
        lr = u.adjust_learning_rate(args, optimizer, epoch)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        # tq.set_description('epoch %d, lr %f' % (epoch, lr))
        tq.set_description('fold %d,epoch %d, lr %f' % (int(1), epoch, lr))
        loss_record = []
        train_loss = 0.0

        # for i, (data, label, edge) in enumerate(dataloader_train):
        # for i, (data, label, edge, d, path) in enumerate(dataloader_train):
        for i, (data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
                # edge = edge.cuda()  #
                # d = d.cuda()

            optimizer.zero_grad()

            # main_out = model(data)  # normal
            main_out, edge_out = model(data)
            # main_out, edge_out = model(data, d)  # 交互式

            # --------------------------Loss----------------------------- #
            # loss_edge = criterion[3](edge_out, edge)
            loss_img = F.binary_cross_entropy_with_logits(main_out, label, weight=None) + criterion[1](main_out, label)

            loss = loss_img
            # loss = 0.2*loss_img + 1.8*loss_edge
            # ----------------------------------------------------------- #

            loss.backward()

            optimizer.step()
            tq.update(args.batch_size)
            train_loss += loss.item()  # item()将一个零维张量转换成浮点数
            tq.set_postfix(loss='%.6f' % (train_loss/(i+1)))  # 进度条右边信息
            step += 1
            if step % 10 == 0:
                writer.add_scalar('Train/loss_step_{}'.format(1), loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('Train/loss_epoch_{}'.format(1), float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        if epoch % args.validation_step == 0:
            Dice, Acc, jaccard, Sensitivity, Specificity= val(args, model, dataloader_val)
            writer.add_scalar('Valid/Dice_val_{}'.format(1), Dice, epoch)
            writer.add_scalar('Valid/Acc_val_{}'.format(1), Acc, epoch)
            writer.add_scalar('Valid/Jac_val_{}'.format(1), jaccard, epoch)
            writer.add_scalar('Valid/Sen_val_{}'.format(1), Sensitivity, epoch)
            writer.add_scalar('Valid/Spe_val_{}'.format(1), Specificity, epoch)

            is_best = Dice > best_pred
            best_pred = max(best_pred, Dice)
            checkpoint_dir_root = args.save_model_path
            # checkpoint_dir = os.path.join(checkpoint_dir_root)
            checkpoint_dir = os.path.join(checkpoint_dir_root, str(1))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_latest = os.path.join(checkpoint_dir, 'checkpoint_latest.pth.tar')
            u.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dice': best_pred,
                    }, best_pred, epoch, is_best, checkpoint_dir, filename=checkpoint_latest)


# 测试
def eval(model, dataloader, args):
    print('start test!')
    with torch.no_grad():
        model.eval()
        # print(model)
        tq = tqdm.tqdm(total=len(dataloader)*args.batch_size)
        tq.set_description('test')

        # *********************************************计算性能指标*********************************************
        total_dice = []
        total_precision = []
        total_jaccard = []
        total_Sensitivity = []
        total_Specificity = []
        start = time.time()
        # for i, (data, label, edge_path, d, label_path) in enumerate(dataloader):
        for i, (data, label, label_path) in enumerate(dataloader):
        # for i, (data, label) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                # d = d.cuda()
                label = label.cuda()

            # ---------------------------------------------- #
            # main_out= model(data, d)
            # main_out = model(data)
            main_out, edge_out = model(data)

            Dice, Acc, Jaccard, Sensitivity, Specificity= u.eval_single_seg(main_out, label)
            total_dice += Dice
            total_precision += Acc
            total_jaccard += Jaccard
            total_Sensitivity += Sensitivity
            total_Specificity += Specificity
            dice = sum(total_dice) / len(total_dice)
            acc = sum(total_precision) / len(total_precision)
            jac = sum(total_jaccard) / len(total_jaccard)
            sen = sum(total_Sensitivity) / len(total_Sensitivity)
            spe = sum(total_Specificity) / len(total_Specificity)
            # #
            main_out = torch.round(torch.sigmoid(main_out)).byte()
            pred_seg = main_out.data.cpu().numpy() * 255
            for index, item in enumerate(label_path):
                # print(item)
                save_img_path = item.replace('labels', r'iteractive\cedi')
                if not os.path.exists(os.path.dirname(save_img_path)):
                    os.makedirs(os.path.dirname(save_img_path))
                # print(item)
                # img = Image.open(item)
                # size = img.size()
                size = (Image.open(item)).size
                img = Image.fromarray(pred_seg.squeeze(), mode='L')
                img = img.resize(size)
                img.save(save_img_path)
                tq.set_postfix(str=str(save_img_path))
        end = time.time()
        print("测试时间", end - start)
        print('Dice:', dice)
        print('Acc:', acc)
        print('Jac:', jac)
        print('Sen:', sen)
        print('Spe:', spe)

        tq.close()


# foot segmentation
def main(mode='train', args=None, writer=None):
    # create dataset and dataloader
    dataset_path = os.path.join(args.data, args.wound)

    dataset_train = LinearLesion(dataset_path, scale=(args.crop_height, args.crop_width), mode='train')
    dataloader_train = DataLoader(  # 迭代器
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    dataset_val = LinearLesion(dataset_path, scale=(args.crop_height, args.crop_width), mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=len(args.cuda.split(',')),
        # the default is 1(the number of gpu), you can set it to what you want
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    dataset_test = LinearLesion(dataset_path, scale=(args.crop_height, args.crop_width), mode='test')
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=len(args.cuda.split(',')),
        # the default is 1(the number of gpu), you can set it to what you want
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda  # 选择GPU 0

    # load model
    model_all = {'UNet': UNet(in_channels=args.input_channel, n_classes=args.num_classes),
                 'CPFNet': CPFNet(),
                 'CE_Net': CE_Net(),
                 'FANet': FANet(),
                 'ResNet34': ResNet34(),
                 'AttU_Net': AttU_Net(),
                 'CSNet': CSNet(),
                 'PSPNet': PSPNet(),
                 'DANet': DANet(),
                 # 'DeepLabv3_plus': DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=True, _print=False)
                 'MobileNetV2': DeepLab(num_classes=1),
                 }

    model = model_all[args.net_work]
    cudnn.benchmark = True  # 增加运行效率
    # model._initialize_weights()
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    # load pretrained model if exists
    if args.pretrained_wound_model_path and mode == 'test':
        print("=> loading pretrained model '{}'".format(args.pretrained_wound_model_path))  # 字符串格式化
        checkpoint_skin = torch.load(args.pretrained_wound_model_path)
        model.load_state_dict(checkpoint_skin['state_dict'])
        print('Done!')

    # 优化器SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Loss
    criterion_aux = nn.BCEWithLogitsLoss(weight=None)  # 包括了 Sigmoid 层和BCELoss 层，比使用一个简单的Sigmoid和一个BCELoss更稳定
    criterion_main = LS.DiceLoss()
    edgeloss = LS.EdgeLoss()
    focalloss = LS.FocalLoss()
    criterion = [criterion_aux, criterion_main, edgeloss, focalloss]

    if mode == 'train':
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer)
    if mode == 'test':
        eval(model, dataloader_test, args)


if __name__ == '__main__':
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为当前GPU设置随机种子
    cudnn.deterministic = True

    args = DefaultConfig()
    modes = args.mode

    if modes == 'train':
        comments = os.getcwd().split('/')[-1]  # 当前工作目录 最后一个/之后 (UNet)
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')  # 格式化时间(简写:月天_时分秒)
        # socket.gethostname()获取本地主机名
        log_dir = os.path.join(args.log_dirs, comments+'_'+current_time + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)  # 可视化
        # for i in range(args.k_fold):  # 0~4
        #     main(mode='train', args=args, writer=writer, k_fold=int(i+1))  # i：1~5
        main(mode='train', args=args, writer=writer)
    elif modes == 'test':
         main(mode='test', args=args, writer=None)
