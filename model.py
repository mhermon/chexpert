from torch import nn

class MultiLabelClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MultiLabelClassification, self).__init__()
        
        # create separate classifiers for our outputs
        self.classpred = nn.Sequential(
            nn.Linear(num_feature, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_class) 
            )
        
        self.classpred2 = nn.Sequential(
            nn.Linear(num_feature, 256),
            nn.ReLU(),
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, num_class)
            )
        
        self.classifier = nn.Linear(num_feature, num_class)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.classifier(x)
        out = self.sigmoid(out)
        return out