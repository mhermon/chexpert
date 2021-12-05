from torch.utils.data import Dataset
from torch import is_tensor, from_numpy
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import numpy as np
import random

class CheXpertDataset(Dataset):
    def __init__(self, X, y, scale_X=None, smooth="smart", class_index=-1):

        # Smooth labels
        if smooth == "ones":
            y[y == -1] = 1
        elif smooth == "ones-lsr":
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if y[i,j] == -1:
                        y[i,j] = random.uniform(.55, .85)
        elif smooth == "zeros":
            y[y == -1] = 0
            y[y == -1] = 0
        elif smooth == 'zeros-lsr':
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if y[i,j] == -1:
                        y[i,j] = random.uniform(0, .3)
        elif smooth == "smart":
            y[:, 1][y[:, 1] == -1] = 1
            y[:, 3][y[:, 3] == -1] = 1
            y[:, 0][y[:, 0] == -1] = 0
            y[:, 2][y[:, 2] == -1] = 0
            y[:, 4][y[:, 4] == -1] = 0
        else:
            raise Exception('Please provide a valid smoothing technique')
            
        # Limit to chosen class
        if class_index != -1:
            y = y[:, class_index]
            y = np.reshape(y, (len(y), 1))
            
        # Calculate imratio
        imratio_list = []
        for i in range(y.shape[1]):
            neg = np.count_nonzero(y[:,i] == 0)
            pos = np.count_nonzero(y[:,i] == 1)
            imratio = pos/(pos+neg)
            imratio_list.append(imratio)
        self.imratio_list = imratio_list
        self.imratio = np.mean(imratio_list)
    
        # Scale data and convert to Tensors
        if not is_tensor(X):
            if scale_X == "standardize":
                print("Standardizing")
                X = StandardScaler().fit_transform(X)
            elif scale_X == "normalize":
                print("Normalizing")
                X = Normalizer().fit_transform(X)
            elif scale_X == "rescale":
                print("Rescaling")
                X = MinMaxScaler().fit_transform(X)
            self.X = from_numpy(X)
        if not is_tensor(y):
            self.y = from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def imbalance_ratio(self):
        return self.imratio
