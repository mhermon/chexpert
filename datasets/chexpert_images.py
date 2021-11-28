import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import cv2
from PIL import Image
import pandas as pd
from glob import glob

class CheXpertImageDataset(Dataset):
    '''
    Reference: 
        Large-scale Robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification
        Zhuoning Yuan, Yan Yan, Milan Sonka, Tianbao Yang
        International Conference on Computer Vision (ICCV 2021)
    '''
    def __init__(self, 
                 labels, 
                 images,
                 image_size=320,
                 class_index=0,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 transforms= None,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion'],
                 mode='train'):
    
        # load data from csv
        self.df = pd.DataFrame(labels, columns=train_cols)
            
        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print ('Upsampling %s...'%col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_listIm, axis=0)


        # impute missing values 
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)  
                self.df[col].fillna(0, inplace=True) 
            elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)
        
        self._num_images = len(self.df)
        
        # 0 --> -1
        if flip_label and class_index != -1: # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)   
            
        # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        if class_index == -1: # 5 classes
            if verbose:
                print ('Multi-label mode: True, Number of classes: [%d]'%len(train_cols))
                print ('-'*30)
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:       # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()
        
        self.mode = mode
        self.class_index = class_index
        self.image_size = image_size
        self.transforms = transforms
        
        self._images_list = images
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()
    
        
        if True:
            if class_index != -1:
                if flip_label:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[-1]+self.value_counts_dict[1])
                    if verbose:
                        print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                        print ('-'*30)
                else:
                    print(self.value_counts_dict)
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[0]+self.value_counts_dict[1])
                    if verbose:
                        print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[0] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                        print ('-'*30)
            else:
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    imratio = self.value_counts_dict[class_key][1]/(self.value_counts_dict[class_key][0]+self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    if verbose:
                        #print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(select_col, class_key, imratio ))
                        print ()
                        #print ('-'*30)
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
         
                
            
    @property        
    def class_counts(self):
        return self.value_counts_dict
    
    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)
       
    @property  
    def data_size(self):
        return self._num_images 
    
    def image_augmentation(self, image):
        img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)]) # pytorch 3.7: fillcolor --> fill
        image = img_aug(image)
        return image
    
    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):

        image = cv2.imread(self._images_list[idx], 0)
        image = Image.fromarray(image)
        if self.mode == 'train' :
            if self.transforms is None:
                image = self.image_augmentation(image)
            else:
                image = self.transforms(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)  
        image = image/255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ =  np.array([[[0.229, 0.224, 0.225]  ]]) 
        image = (image-__mean__)/__std__
        image = image.transpose((2, 0, 1)).astype(np.float32)
        if self.class_index != -1: # multi-class mode
            label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        return image, label