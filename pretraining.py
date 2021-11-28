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
from libauc.losses import AUCMLoss, CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from libauc.models import DenseNet121
import matplotlib.pyplot as plt
from torchvision import transforms

parser = ap.ArgumentParser()
parser.add_argument('-d', '--image_dir', type=Path, required=True)
parser.add_argument('-l', '--labels', type=float, required=True)
parser.add_argument('--seed', type=float, required=True)
parser.add_argument('-val', '--val_prop', type=float, required=True)
parser.add_argument('-o', '--output_dir', type=Path, required=True)
parser.add_argument('--dist',
                    default='copy',
                    const='copy',
                    nargs='?',
                    choices=('copy', 'all'),
                    help='Provide a distribution to sample as')

p = parser.parse_args()
image_dir = p.image_dir
label_file = p.labels
seed = p.seed


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_all_seeds(seed)
extra_valid_labels = np.load(label_file)
extra_valid_images = glob(image_dir)

# Split into train and validation
train_images, val_images, train_labels, val_labels = train_test_split(extra_valid_images, extra_valid_labels_df,
                                                  test_size=0.2, random_state=SEED)















