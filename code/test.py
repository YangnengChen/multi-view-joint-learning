import os


import random
import numpy as np
import torch
import cv2
import timm
import pandas as pd
import torchvision.transforms as transform
from sklearn import model_selection
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from timm.models import create_model
from models.model import ce_loss,ETMC
from dataset import CustomDataset, MultiviewImgDataset, SingleimgDataset
from sklearn.metrics import classification_report
import models.create_models as create

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


GPU=[0,1,2]
def model_parallel(model):
    device_ids = [i for i in GPU]
    model = nn.DataParallel(model, device_ids=device_ids)

    return model


def model_forward(i_epoch, ce_loss,alpha,alpha_a, tgt):
    n_classes=5
    annealing_epoch=2
    ce_losses=[ce_loss(tgt, alpha[i], n_classes, i_epoch, annealing_epoch) for i in range(NUM_VIEW)]
    loss = sum(ce_losses) + \
           ce_loss(tgt, alpha_a, n_classes, i_epoch, annealing_epoch)
    return loss
def test_model2(netname, model, test_loader, dataset_size, criterion, device):
    n_class = 5
    print('dataset_size', dataset_size)
    since = time.time()
    # roc matrix
    if not CROSS:
        roc_matrix = pd.DataFrame(columns=['label', '0', '1', '2', '3', '4'])
    else:
        roc_matrix = pd.DataFrame(columns=['样本id','0','1','2','3','4','分级结果','level'])

    # fusion matrix
    total_outs = torch.tensor([])
    total_labels = torch.tensor([])
    total_img_ids = []
    FM = np.zeros((n_class, n_class))
    tp = [0] * n_class
    tn = [0] * n_class
    fp = [0] * n_class
    fn = [0] * n_class
    precision = [0] * n_class
    recall = [0] * n_class
    specificity = [0] * n_class
    f1 = [0] * n_class
    valid_epoch_loss = 0.0
    valid_epoch_acc = 0.0

    running_loss = 0.0
    running_corrects = 0
    model.eval()
    # 收集数据标签
    all_labels = []
    
    valid_bar = tqdm(test_loader)
    for i, (img, label,img_id) in enumerate(valid_bar):
        for i in label:
            all_labels.append(i)
        img = img.to(device)
        label_cuda = label.to(device)
        B, V, C, H, W = img.size()

        # img = img.view(-1, C, H, W)
        img = img.to(device, non_blocking=True)

        # b,v = label.shape
        # label=label.view(b*v)
        if USING_ETMC:
            alpha,alpha_a=model(img)
            loss=model_forward(i_epoch=20,ce_loss=ce_loss,alpha=alpha,alpha_a=alpha_a,tgt=label_cuda)
            outputs=alpha_a.cpu()
        else:
            with torch.no_grad():
                output, t_vis = model(img)
            outputs = output[0].cpu()
            # _, pred = torch.max(outputs.data, 1)
            loss = criterion(outputs, label)
        
        total_labels = torch.cat((total_labels, label), dim=0) if total_labels.size(0) > 0 else label
        total_outs = torch.cat((total_outs, outputs.detach()), dim=0) if total_outs.size(0) > 0 else outputs.detach()
        
        for id in img_id:
            total_img_ids.append(id)
        
        running_loss += loss.item()
        
    # # 统计数据分布
    label_counts = [0,0,0,0,0]
    for label in all_labels:
        label_counts[label] += 1

    # 打印数据分布
    print("数据分布：")
    for label in [0,1,2,3,4]:
        print(f"类别 {label}: {label_counts[label]} 个样本")
    print(len(total_img_ids))    
    preds = torch.argmax(total_outs, dim=1)
    outputs_softmax = torch.softmax(total_outs, 1)
    labels = total_labels
    img_ids = total_img_ids
    # print(classification_report(labels,preds))
    cls_statis = [torch.sum(labels == 0), torch.sum(labels == 1), sum(labels == 2), sum(labels == 3), sum(labels == 4)]

    running_corrects += torch.sum(preds == labels)
    # running_corrects+=(output1.argmax(dim=1) == label).sum()

    for batch_i in range(len(labels)):

        # save roc matrix
        if not CROSS:
            roc_df = {'label': labels[batch_i],
                  '0': outputs_softmax[batch_i][0],
                  '1': outputs_softmax[batch_i][1],
                  '2': outputs_softmax[batch_i][2],
                  '3': outputs_softmax[batch_i][3],
                  '4': outputs_softmax[batch_i][4]
                  }
        else:
            roc_df = {
                  '样本id': img_ids[batch_i],
                  '0': outputs_softmax[batch_i][0].item(),
                  '1': outputs_softmax[batch_i][1].item(),
                  '2': outputs_softmax[batch_i][2].item(),
                  '3': outputs_softmax[batch_i][3].item(),
                  '4': outputs_softmax[batch_i][4].item(),
                  '分级结果': preds[batch_i].item(),
                  'level': labels[batch_i].item(),
                  }        
        # roc_matrix = roc_matrix.append(roc_df, ignore_index=True)
        roc_matrix = pd.concat([roc_matrix, pd.DataFrame([roc_df])], ignore_index=True)

        # fusion matrix
        predict_label = preds[batch_i]
        true_label = labels[batch_i]
        # FM[predict_label][true_label] = FM[predict_label][true_label] + 1
        FM[true_label][predict_label] = FM[true_label][predict_label] + 1

        for label in range(n_class):
            p_or_n_from_pred = (label == preds[batch_i])
            p_or_n_from_label = (label == labels[batch_i])

            if p_or_n_from_pred == 1 and p_or_n_from_label == 1:
                tp[label] += 1
            if p_or_n_from_pred == 0 and p_or_n_from_label == 0:
                tn[label] += 1
            if p_or_n_from_pred == 1 and p_or_n_from_label == 0:
                fp[label] += 1
            if p_or_n_from_pred == 0 and p_or_n_from_label == 1:
                fn[label] += 1

    # each class test results
    for label in range(n_class):
        precision[label] = tp[label] / (tp[label] + fp[label] + 1e-8)
        recall[label] = tp[label] / (tp[label] + fn[label] + 1e-8)
        specificity[label] = tn[label] / (tn[label] + fp[label] + 1e-8)
        f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label] + 1e-8)

        print('Class {}: \t Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, F1:{:.4f}'.format(
            label, precision[label], recall[label], specificity[label], f1[label]))
        fileHandle = open(netname + '_test_result.txt', 'a')
        fileHandle.write('Class {}: \t Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, F1:{:.4f}\n'.format(
            label, precision[label], recall[label], specificity[label], f1[label]))
        fileHandle.close()

        # save Fusion Matric
    print('\nFusion Matrix:')
    print(FM)
    fileHandle = open(netname + '_test_result.txt', 'a')
    fileHandle.write('\nFusion Matrix:\n')
    for f_i in FM:
        fileHandle.write(str(f_i) + '\r\n')
    fileHandle.close()

    # save roc data
    if not CROSS:
        roc_matrix.to_csv(netname + '_roc_data.csv', encoding='gbk')
    else:
        roc_matrix.to_csv(netname + '_交叉验证.csv', encoding='gbk')

    # calculate the Kappa
    pe0 = (tp[0] + fn[0]) * (tp[0] + fp[0])
    pe1 = (tp[1] + fn[1]) * (tp[1] + fp[1])
    pe2 = (tp[2] + fn[2]) * (tp[2] + fp[2])
    pe3 = (tp[3] + fn[3]) * (tp[3] + fp[3])
    pe4 = (tp[4] + fn[4]) * (tp[4] + fp[4])
    pe = (pe0 + pe1 + pe2 + pe3 + pe4) / (dataset_size * dataset_size)
    pa = (tp[0] + tp[1] + tp[2] + tp[3] + tp[4]) / dataset_size
    kappa = (pa - pe) / (1 - pe)

    # overall test results
    test_epoch_loss = running_loss / dataset_size
    test_epoch_acc = running_corrects / dataset_size
    overall_precision = sum([cls_statis[i] * p for i, p in enumerate(precision)]) / sum(cls_statis)
    overall_recall = sum([cls_statis[i] * r for i, r in enumerate(recall)]) / sum(cls_statis)

    overall_specificity = sum([cls_statis[i] * s for i, s in enumerate(specificity)]) / sum(cls_statis)
    overall_f1 = sum([cls_statis[i] * f for i, f in enumerate(f1)]) / sum(cls_statis)

    elapsed_time = time.time() - since
    print(
        'Test loss: {:.4f}, acc: {:.4f}, kappa: {:.4f}, avg_precision: {:.4f},avg_recall: {:.4f},avg_specificity: {:.4f},avg_f1: {:.4f},Total elapsed time: {:.4f} '.format(
            test_epoch_loss, test_epoch_acc, kappa, overall_precision, overall_recall, overall_specificity, overall_f1,
            elapsed_time))
    fileHandle = open(netname + '_test_result.txt', 'a')
    fileHandle.write(
        'Test loss: {:.4f}, acc: {:.4f}, kappa: {:.4f}, all_precision: {:.4f},all_recall: {:.4f}, all_specificity: {:.4f}, all_f1: {:.4f}, Total elapsed time: {:.4f} \n'.format(
            test_epoch_loss, test_epoch_acc, kappa, overall_precision, overall_recall, overall_specificity, overall_f1,
            elapsed_time))
    fileHandle.close()
    return (test_epoch_loss, test_epoch_acc)


if __name__ == '__main__':
    IMAGE_SIZE=544
    transform_test = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.ToTensor(),
        transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    DATA_PATH = "/data/cyn/data/MFIDDR"

    TEST_PATH = "/data/cyn/data/MFIDDR/test/"
    BATCH_SIZE = 8
    MODELPATH = "/data/cyn/model/nfnet_MVCINN/ETMC.pth"
    checkpoint = torch.load(MODELPATH)
    DEPTH = 12
    HEAD = 9
    BATCH_SIZE = 4
    PARALLEL=True
    VALID=False
    
    CROSS=False
    
    USING_DECODER=True
    DECODER_EMBEDDING=3072
    NUM_LAYERS_DECODER=8
    ABLATION_FUSION = False
    CONCAT=False
    IS_WEIGHTED_JOINT=False
    FUSE_VIEW= False
    SK_NET =True
    USING_ETMC= True
    NUM_VIEW=4
    M=4
    R=8
    L=32
    if USING_ETMC:
        model=ETMC()
    else:
        if isinstance(checkpoint, dict):
            model = create.my_MVCINN(
            pretrained=False,
            num_classes=5,
            pre_Path = 'weights/final_0.8010.pth',
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
            )
        else:
            model = checkpoint
    model=model_parallel(model)
    weights_dict = torch.load(MODELPATH, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print(model.load_state_dict(weights_dict, strict=False))
    test_csv_path = os.path.join(DATA_PATH, 'test_fourpic_label.csv')
    test_df = pd.read_csv(test_csv_path)
    test_dataset = MultiviewImgDataset(TEST_PATH, test_df, transform=transform_test)

    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32, pin_memory=False)

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    if not USING_ETMC:
        model_name=f'MVCINN_test_Nfnet_conv2d_SKnet_{SK_NET}_M_{M}_r{R}_L_{L}_is_weighted_joint_{IS_WEIGHTED_JOINT}_fuse_view_{FUSE_VIEW}_concat_{CONCAT}_ablation_fusion_{ABLATION_FUSION}_using_decoder_{USING_DECODER}_decoder_embedding_{DECODER_EMBEDDING}_num_layers_decoder_{NUM_LAYERS_DECODER}_best_acc'
    else:
        model_name=f'ETMC'
    test_model2(model_name, model, test_loader, len(test_dataset),
                nn.CrossEntropyLoss(reduction='sum'), device)