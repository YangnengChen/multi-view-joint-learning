

import os


import random
import numpy as np
import torch
import cv2
import timm
import pandas as pd
import copy
import matplotlib.pyplot as plt
import torchvision.transforms as transform
from sklearn import model_selection
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
import models.create_models as create
from tqdm import tqdm

from dataset import CustomDataset, MultiviewImgDataset, SingleimgDataset
from models.FLoss import FocalLoss
from sklearn.model_selection import train_test_split
from ml_decoder import MLDecoder
#cyn
import argparse
import datetime
import json

import time
from pathlib import Path

import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm


import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from models.model import ce_loss,ETMC


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4"


GPU=[0,1,2]
def model_parallel(model):
    device_ids = [i for i in GPU]
    model = nn.DataParallel(model, device_ids=device_ids)

    return model


def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset


    
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def model_forward(i_epoch, ce_loss,alpha,alpha_a, tgt):
    n_classes=5
    annealing_epoch=2
    ce_losses=[ce_loss(tgt, alpha[i], n_classes, i_epoch, annealing_epoch) for i in range(NUM_VIEW)]
    loss = sum(ce_losses) + \
           ce_loss(tgt, alpha_a, n_classes, i_epoch, annealing_epoch)
    return loss

def main(model_name,N_EPOCHS=100,LR = 0.0001,depth=12,head=9):
    # model = ViT(num_classes=5, pretrained=True)
    # model = Deit(num_classes=5, pretrained=True)
    # model = ResNet50(num_classes=5 , heads=4)
    print(model_name)
    if USING_ETMC:
        model=ETMC()
    else:
        model = create.my_MVCINN(
            pretrained=False,
            num_classes=5,
            pre_Path = 'weights/final_0.8010.pth',
            depth=depth,
            num_heads=head,
            # embed_dim=768,
            drop_rate=0.2,
            drop_path_rate=0.2,
            using_decoder=USING_DECODER,
            decoder_embedding=DECODER_EMBEDDING,
            num_layers_decoder=NUM_LAYERS_DECODER,
            ablation_fusion=ABLATION_FUSION,
            concat= CONCAT,
            fuse_view=FUSE_VIEW,
            is_weighted_joint=IS_WEIGHTED_JOINT,
            SK_net=SK_NET,
            M=M, r=R, L=L,
            num_of_groups=NUM_GROUPS,
        )



    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    # scheduler = CosineAnnealingLR(optimizer,T_max=5)
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=2)
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    inter_val = 2

    best_model_acc = copy.deepcopy(model)
    best_model_loss = copy.deepcopy(model)
    last_model = model
    
    if PARALLEL:
        model=model_parallel(model)
    
    print(model.to(device))
    

    best_acc = 0
    best_loss = float('inf')
    best_test = 0
    for epoch in range(N_EPOCHS):

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0

        model.train()
        train_bar = tqdm(train_loader)

        for i, (img, label,img_id) in enumerate(train_bar):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            B, V, C, H, W = img.size()
            
            # mixup
            alpha=0.2
            lam = np.random.beta(alpha,alpha)
            index = torch.randperm(B).cuda()
            imgs_mix = lam*img + (1-lam)*img[index]
            label_a,label_b = label,label[index]
            

            # imgs_mix = imgs_mix.view(-1, C, H, W)

            # b,v = label.shape
            # label=label.view(b*v)
            # print(imgs_mix.shape)
            optimizer.zero_grad()
            if USING_ETMC:
                alpha,alpha_a=model(imgs_mix)
                loss=model_forward(epoch,ce_loss=ce_loss,alpha=alpha,alpha_a=alpha_a,tgt=label)
                output=alpha_a
            else:
                output,_ = model(imgs_mix)
                output = output[0]
                loss = lam*criterion(output, label_a) + (1-lam)*criterion(output, label_b)
            # output = output[1]+output[0]
            

            # print(label.shape,output.shape)
            


            
            loss.backward()
            train_epoch_acc += (output.argmax(dim=1) == label).sum()
            train_epoch_loss += loss.item()


            scheduler.step(epoch + i / len(train_loader))
            optimizer.step()



        train_loss_mean = train_epoch_loss / len(train_loader)
        train_acc_mean = train_epoch_acc / (len(train_dataset) * NUM_VIEW)


        train_loss.append(train_loss_mean)
        train_acc.append(train_acc_mean.cpu())

        print('{} train loss: {:.3f} train acc: {:.3f}  lr:{}'.format(epoch,train_loss_mean,
                                                                        train_acc_mean,
                                                                        optimizer.param_groups[-1]['lr']))
        if (epoch + 1) % inter_val == 0:
            val_acc_mean,  val_loss_mean = val_model(model, test_loader, criterion,device)

            if val_acc_mean > best_acc:
                model.cpu()
                best_model_acc = copy.deepcopy(model.state_dict() if PARALLEL else model.module.state_dict())
                model.to(device)
                best_acc = val_acc_mean
            if val_loss_mean < best_loss:
                model.cpu()
                best_model_loss = copy.deepcopy(model.state_dict() if PARALLEL else model.module.state_dict())
                model.to(device)
                best_loss = val_loss_mean
            
            
            
            valid_loss.append(val_loss_mean)
            valid_acc.append(val_acc_mean.cpu())


        # test_acc_mean = testModel(model,test_loader,len(test_dataset),device)
        # if test_acc_mean>best_test:
        #     best_test = test_acc_mean

    print("best val acc:", best_acc)
    if best_test != 0: print("best test acc:", best_test)
    torch.save(best_model_acc, os.path.join(SAVE_PT_DIR,'best_acc-{}-{:.4f}.pth'.format(model_name,best_acc)))
    print("best val loss:", best_loss)
    torch.save(best_model_loss, os.path.join(SAVE_PT_DIR,'best_loss-{}-{:.4f}.pth'.format(model_name,best_loss)))

    # torch.save(model, os.path.join(SAVE_PT_DIR,'last2.pt'))
    print("model saved at weights")




def val_model(model, valid_loader, criterion, device):
    valid_epoch_loss = 0.0
    valid_epoch_acc = 0.0

    model.eval()
    valid_bar = tqdm(valid_loader)
    for i, (img, label,img_id) in enumerate(valid_bar):
        img = img.to(device)
        label = label.to(device)

        B, V, C, H, W = img.size()
        # img = img.view(-1, C, H, W)
        img = img.to(device, non_blocking=True)

        # b,v = label.shape
        # label=label.view(b*v)

        with torch.no_grad():
            output,_ = model(img)
        output = output[0]
        
        
        loss = criterion(output, label)
        valid_epoch_loss += loss.item()
        valid_epoch_acc += (output.argmax(dim=1) == label).sum()


    val_acc_mean = valid_epoch_acc / (len(valid_dataset) * NUM_VIEW)
    val_loss_mean = valid_epoch_loss / len(valid_loader)

    
    print('valid loss: {:.3f} valid acc: {:.3f}'.format(val_loss_mean,val_acc_mean))
    return val_acc_mean, val_loss_mean


if __name__ == '__main__':
    seed_everything(1001)
    
    # general global variables
    DATA_PATH = "/data/cyn/data/MFIDDR"
    TRAIN_PATH = "/data/cyn/data/MFIDDR/train/"
    TEST_PATH = "/data/cyn/data/MFIDDR/test/"
    SAVE_IMG_DIR = 'imgs'
    SAVE_PT_DIR = '/data/cyn/model/nfnet_MVCINN'
    NUM_VIEW = 1
    IMAGE_SIZE = 544
    LR = 0.00001
    N_EPOCHS = 20
    DEPTH = 12
    HEAD = 9
    BATCH_SIZE = 8
    PARALLEL=True
    VALID=False
    
    USING_DECODER=True
    DECODER_EMBEDDING=3072
    NUM_LAYERS_DECODER=4
    NUM_GROUPS=-1
    
    ABLATION_FUSION = False
    CONCAT=False
    IS_WEIGHTED_JOINT=False
    FUSE_VIEW= False
    SK_NET =True
    USING_ETMC= True
    M=4
    R=8
    L=32
    train_csv_path = os.path.join(DATA_PATH, 'train_fourpic_label.csv')
    assert os.path.exists(train_csv_path), '{} path is not exists...'.format(train_csv_path)
    test_csv_path = os.path.join(DATA_PATH, 'test_fourpic_label.csv')
    df_test = pd.read_csv(test_csv_path)

    train_data = pd.read_csv(train_csv_path)
    train_data.head()
    df_train, df_val = train_test_split(train_data, test_size=0.2,random_state=1)

    transform_train = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.RandomHorizontalFlip(p=0.3),
        transform.RandomVerticalFlip(p=0.3),
        transform.RandomResizedCrop(IMAGE_SIZE),
        transform.ToTensor(),
        transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_valid = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.ToTensor(),
        transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((544, 544)),
        transform.ToTensor(),
        transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if VALID:
        train_dataset = MultiviewImgDataset(TRAIN_PATH, df_train, transform=transform_train)
        valid_dataset = MultiviewImgDataset(TEST_PATH, df_val, transform=transform_test)
        test_dataset = MultiviewImgDataset(TEST_PATH, df_test, transform=transform_test)
    else:
        train_dataset = MultiviewImgDataset(TRAIN_PATH, train_data, transform=transform_train)
        valid_dataset = MultiviewImgDataset(TEST_PATH, df_test, transform=transform_test)
        test_dataset = MultiviewImgDataset(TEST_PATH, df_test, transform=transform_test)
    
    
    

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    date=datetime.datetime.now().strftime('%m-%d-%H:%M')
    if not USING_ETMC:
        model_name=f'MVCINN_Nfnet_conv2d_SKnet_{SK_NET}_M_{M}_r{R}_L_{L}_is_weighted_joint_{IS_WEIGHTED_JOINT}_fuse_view_{FUSE_VIEW}_concat_{CONCAT}_ablation_fusion_{ABLATION_FUSION}_using_decoder_{USING_DECODER}_decoder_embedding_{DECODER_EMBEDDING}_num_layers_decoder_{NUM_LAYERS_DECODER}_num_of_groups_{NUM_GROUPS}_{date}_{LR}_{N_EPOCHS}'
    else:
        model_name=f'ETMC_{date}'
    main(model_name,N_EPOCHS = N_EPOCHS,LR = LR,depth=DEPTH,head=HEAD)