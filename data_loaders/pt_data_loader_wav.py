 #This library contains classes and functions to load FRPs and apply custom data 
 #augmentation operations to models implemented in PyTorch.

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import os

class Dataset_Spec_from_wav(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dir_img, list_IDs, list_labels, unique_labels, train=True, multi_instance=False,scale=(0.05, 0.1), noise_level=0.05):
        'Initialization'
        self.dir = dir_img
        self.list_labels = list_labels
        self.list_IDs = list_IDs
        self.unique_labels = unique_labels
        self.TrainFlag = train
        self.multi_instance = multi_instance
        self.scale = scale
        self.noise_level = noise_level
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.load_image(self.dir, ID)
        y = torch.as_tensor(self.unique_labels.get(self.list_labels[index]))
        y = torch.squeeze(y)
        y = y.type(torch.LongTensor)

        return X, y
    
    def Transformations_train(self,img):
        preprocess = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.ToPILImage(),
            transforms.Resize(512),
            AddGaussianNoise(std=self.noise_level),
            transforms.ToTensor(),#normaliza a [0,1]
            Time_Masking_layer(p=0.7, scale=self.scale),
            Freq_Masking_layer(p=0.7, scale=self.scale),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        image_tensor = preprocess(img)
        return image_tensor

    def Transformations_test(self,img):
        preprocess = transforms.Compose([
            transforms.Resize(512),
            AddGaussianNoise(std=self.noise_level),
            transforms.ToTensor(),#normaliza a [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        image_tensor = preprocess(img)
        return image_tensor

    def load_image(self, img_path,ID):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        if self.multi_instance:
            file_exp = ID.split('_')
            n_sp = file_exp[1]
            axis = file_exp[3]
            image = cv2.imread(img_path + file_exp[0] + '_' + n_sp + os.sep + 'Axis_' + axis + os.sep + file_exp[4] + '.png')
        else:
            image = cv2.imread(img_path + ID + '.png')
        # resize image
        dsize = (image_size, image_size)
        image = cv2.resize(image, dsize)
        #---------------------------------------------
        image2 = np.copy(image)
        image2[image2>0]=255
        image2 = image2[:,:,0]
        mask = Image.fromarray(image2.astype('uint8'))
        #---------------------------------------------
        img = Image.fromarray(image.astype('uint8'), 'RGB')
        img_adapteq = ImageOps.equalize(img,mask=mask)
        if self.TrainFlag:
            return self.Transformations_train(img_adapteq)
        else:
            return self.Transformations_test(img_adapteq)

class Dataset_Spec_from_wav_CL(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dir_img, list_IDs, list_labels, unique_labels, multi_instance=False, scale=(0.1, 0.2), noise_level=0.05):
        'Initialization'
        self.dir = dir_img
        self.list_labels = list_labels
        self.list_IDs = list_IDs
        self.unique_labels = unique_labels
        self.multi_instance = multi_instance
        self.scale = scale
        self.noise_level = noise_level
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label

        X1 = self.load_image(self.dir, ID)
        X2 = self.load_image(self.dir, ID)
        y = torch.as_tensor(self.unique_labels.get(self.list_labels[index]))
        y = torch.squeeze(y)
        y = y.type(torch.LongTensor)

        return (X1,X2), y
    
    def load_image(self, img_path,ID, multi_instance=False):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        if self.multi_instance:
            file_exp = ID.split('_')
            n_sp = file_exp[1]
            axis = file_exp[3]
            image = cv2.imread(img_path + file_exp[0] + '_' + n_sp + os.sep + 'Axis_' + axis + os.sep + file_exp[4] + '.png')
        else:
            image = cv2.imread(img_path + ID + '.png')
        # resize image
        dsize = (image_size, image_size)
        image = cv2.resize(image, dsize)
        #---------------------------------------------
        image2 = np.copy(image)
        image2[image2>0]=255
        image2 = image2[:,:,0]
        mask = Image.fromarray(image2.astype('uint8'))
        #---------------------------------------------
        img = Image.fromarray(image.astype('uint8'), 'RGB')
        img_adapteq = ImageOps.equalize(img,mask=mask)

        preprocess = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.ToPILImage(),
            transforms.Resize(512),
            AddGaussianNoise(std=self.noise_level),
            transforms.ToTensor(),#normaliza a [0,1]
            Time_Masking_layer(p=0.7, scale=self.scale),
            Freq_Masking_layer(p=0.7, scale=self.scale),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        image_tensor = preprocess(img_adapteq)
        return image_tensor