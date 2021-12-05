import argparse as ap
from pathlib import Path
from model import MultiLabelClassification

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

parser = ap.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--model_path', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
p = parser.parse_args()

test_features = np.load(p.input)
#test_features = StandardScaler().fit_transform(test_features)
test_loader = DataLoader(test_features,
                         batch_size=1,
                         shuffle=False,
                         num_workers=1)
model = MultiLabelClassification(num_feature=1024, num_class=5)
model.load_state_dict(torch.load(p.model_path))
model.eval()

with torch.no_grad():    
    test_pred = []
    for idx, data in enumerate(test_loader):
        test_data = data
        y_pred = model(test_data)
        test_pred.append(y_pred.numpy())
    test_pred = np.concatenate(test_pred)
    np.save(p.output, test_pred)
