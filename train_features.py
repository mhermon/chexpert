import os
from random import randrange
from glob import glob
import argparse as ap

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import CheXpertDataset, CheXpertImageDataset
from model import MultiLabelClassification
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from libauc.losses import AUCMLoss, CrossEntropyLoss, AUCM_MultiLabel
from libauc.optimizers import PESG, Adam
from libauc.models import DenseNet121
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path
import random
from torch.utils.tensorboard import SummaryWriter

parser = ap.ArgumentParser()
parser.add_argument('-tf', '--train_features', type=Path, required=True)
parser.add_argument('-tl', '--train_labels', type=Path, required=True)
parser.add_argument('-vf', '--val_features', type=Path, required=True)
parser.add_argument('-vl', '--val_labels', type=Path, required=True)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.05)
parser.add_argument('--gamma', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--margin', type=float, default=1.0)
parser.add_argument('--decay_factor', type=int, default=10)
parser.add_argument('--class_id', type=int, default=-1)
parser.add_argument('--model_name', required=True)
parser.add_argument('--label_smoothing',
                    default='ones',
                    const='ones',
                    nargs='?',
                    choices=('ones', 'zeros', 'ones-lsr', 'zeros-lsr', 'smart'),
                    help='Provide a label smoothing technique to use')


p = parser.parse_args()
seed = p.seed
epochs = p.num_epochs
smoothing = p.label_smoothing
lr = p.learning_rate 
gamma = p.gamma
weight_decay = p.weight_decay
margin = p.margin
decay_factor = p.decay_factor
class_id = p.class_id
model_name = p.model_name

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_all_seeds(seed)
train_labels = np.load(p.train_labels)
train_features = np.load(p.train_features)
val_labels = np.load(p.val_labels)
val_features = np.load(p.val_features)


train_set = CheXpertDataset(train_features, y=train_labels, scale_X=None, smooth=smoothing, class_index=class_id)
val_set = CheXpertDataset(val_features, y=val_labels, scale_X=None, smooth=smoothing, class_index=class_id)
train_loader = DataLoader(train_set,
                         batch_size=32,
                         shuffle=True,
                         num_workers=2)
val_loader = DataLoader(val_set,
                        batch_size=32,
                        shuffle=False,
                        num_workers=2)

num_features = train_features.shape[1]
num_classes = train_labels.shape[1]

if class_id != -1:
    num_classes = 1

writer = SummaryWriter('./runs/class_1')


# Single class model
if num_classes == 1:
    # model
    model = MultiLabelClassification(num_feature=num_features, num_class=num_classes)
    model = model.cuda()
    imratio = train_set.imratio
    # define loss & optimizer
    Loss = AUCMLoss(imratio=imratio)
    optimizer = PESG(model, 
                     a=Loss.a, 
                     b=Loss.b, 
                     alpha=Loss.alpha,
                     lr=lr, 
                     gamma=gamma, 
                     margin=margin, 
                     weight_decay=weight_decay)

    best_val_auc = 0
    step = 0
    for epoch in range(epochs):
        if epoch > 0:
             optimizer.update_regularizer(decay_factor=decay_factor)
        for idx, data in enumerate(train_loader):
            train_data, train_labels = data
            train_data, train_labels = train_data.cuda(), train_labels.cuda()
            y_pred = model(train_data)
            loss = Loss(y_pred, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 200 == 0:
                writer.add_scalar("Loss/train", loss, step + 1)

            # validation
            if idx % 400 == 0:
                model.eval()
                with torch.no_grad():    
                    test_pred = []
                    test_true = [] 
                    for jdx, data in enumerate(val_loader):
                        test_data, test_label = data
                        test_data = test_data.cuda()
                        y_pred = model(test_data)
                        test_pred.append(y_pred.cpu().detach().numpy())
                        test_true.append(test_label.numpy())

                    test_true = np.concatenate(test_true)
                    test_pred = np.concatenate(test_pred)
                    val_auc =  roc_auc_score(test_true, test_pred) 
                    writer.add_scalar("Validation AUC", val_auc, step + 1)
                    model.train()

                    if best_val_auc < val_auc:
                        print(f'Saved model to {model_name}')
                        torch.save(model.state_dict(), f'./models/{model_name}.pth')
                        best_val_auc = val_auc

                print('Epoch=%s, BatchID=%s, Val_AUC=%.4f, lr=%.4f'%(epoch, idx, val_auc,  optimizer.lr))
                writer.flush()

            step = step + 1

    writer.flush()
    print ('Best Val_AUC is %.4f'%best_val_auc)

else:
    # Multi-Class 
    # model
    imratio = train_set.imratio_list

    model = MultiLabelClassification(num_feature=num_features, num_class=num_classes)
    model = model.cuda()

    # define loss & optimizer
    Loss = AUCM_MultiLabel(imratio=imratio, num_classes=5)
    optimizer = PESG(model, 
                     a=Loss.a, 
                     b=Loss.b, 
                     alpha=Loss.alpha, 
                     lr=lr, 
                     gamma=gamma, 
                     margin=margin, 
                     weight_decay=weight_decay, device='cuda')


    # training
    best_val_auc = 0 
    step = 0
    for epoch in range(5):
        if epoch > 0:
            optimizer.update_regularizer(decay_factor=10)       
        for idx, data in enumerate(train_loader):
            train_data, train_labels = data
            train_data, train_labels  = train_data.cuda(), train_labels.cuda()
            y_pred = model(train_data)
            y_pred = torch.sigmoid(y_pred)
            loss = Loss(y_pred, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 200 == 0:
                writer.add_scalar("Loss/train", loss, step + 1)

            # validation  
            if idx % 400 == 0:
                model.eval()
                with torch.no_grad():    
                    test_pred = []
                    test_true = [] 
                    for jdx, data in enumerate(val_loader):
                        test_data, test_labels = data
                        test_data = test_data.cuda()
                        y_pred = model(test_data)
                        test_pred.append(y_pred.cpu().detach().numpy())
                        test_true.append(test_labels.numpy())

                    test_true = np.concatenate(test_true)
                    test_pred = np.concatenate(test_pred)
                    val_auc_mean =  roc_auc_score(test_true, test_pred) 
                    writer.add_scalar("Validation AUC", val_auc_mean, step + 1)
                    model.train()

                    if best_val_auc < val_auc_mean:
                        best_val_auc = val_auc_mean
                        print(f'Saved model to {model_name}')
                        torch.save(model.state_dict(), f'./models/{model_name}.pth')

                print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc))
                writer.flush()
            step = step + 1

    writer.flush()
    print('Best Val_AUC is %.4f'%best_val_auc)






