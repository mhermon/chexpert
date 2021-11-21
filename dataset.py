from torch.utils.data import Dataset
from torch import is_tensor, from_numpy
from sklearn.preprocessing import StandardScaler

class ChexpertDataset(Dataset):
    def __init__(self, X, y, scale_X=True):
        if not is_tensor(X):
            if scale_X:
                X = StandardScaler().fit_transform(X)
            self.X = from_numpy(X)
        if not is_tensor(y):
            self.y = from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]