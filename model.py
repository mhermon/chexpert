from torch import nn
from libauc.models import DenseNet121

class MultiLabelClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MultiLabelClassification, self).__init__()
        
        # create separate classifiers for our outputs
        self.classpred0 = nn.Sequential(
            nn.Linear(num_feature, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_class) 
            )
        
        self.classpred1 = nn.Sequential(
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
        model = DenseNet121(pretrained=True, last_activation=None, activations='relu', num_classes=5)
        layer = model._modules.get('classifier')
        self.classifier.weight = layer.weight   
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.classpred0(x)
        out = self.sigmoid(out)
        
        return out