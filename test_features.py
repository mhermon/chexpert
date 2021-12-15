import argparse as ap
from pathlib import Path
from model import MultiLabelClassification

import torch
import numpy as np
from torch.utils.data import DataLoader

parser = ap.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--model_paths', nargs='+', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--ensemble', action='store_true')
p = parser.parse_args()
ensemble = p.ensemble
model_paths = p.model_paths
# ASSUMES MODELS ARE IN ORDER OF CLASSES

test_features = np.load(p.input)

if ensemble:
    preds = []
    for model_path in model_paths:
        test_loader = DataLoader(test_features,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1)
        model = MultiLabelClassification(num_feature=1024, num_class=1)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        with torch.no_grad():    
            test_pred = []
            for idx, data in enumerate(test_loader):
                test_data = data
                y_pred = model(test_data)
                test_pred.append(y_pred.numpy())
            test_pred = np.concatenate(test_pred).reshape((-1,1))
            preds.append(test_pred)
    
    preds = np.hstack(preds)
    np.save(p.output, preds)
        
else:
    model_path = model_paths[0]
    test_loader = DataLoader(test_features,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1)
    model = MultiLabelClassification(num_feature=1024, num_class=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():    
        test_pred = []
        for idx, data in enumerate(test_loader):
            test_data = data
            y_pred = model(test_data)
            test_pred.append(y_pred.numpy())
        test_pred = np.concatenate(test_pred)
        np.save(p.output, test_pred)
