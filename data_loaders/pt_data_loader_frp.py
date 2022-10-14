 #This library contains classes and functions to load FRPs and apply custom data 
 #augmentation operations to models implemented in PyTorch.

import torch
from torchvision import transforms
import numpy as np
import os
import cv2
from PIL import Image, ImageOps

image_size = 512

class Dataset_FPR(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dir_img, list_IDs, list_labels, unique_labels, multi_instance=False, scale=0.8,train=True, noise_level=0.05):
        'Initialization'
        self.dir = dir_img
        self.list_labels = list_labels
        self.list_IDs = list_IDs
        self.unique_labels = unique_labels
        self.factor = scale
        self.multi_instance = multi_instance
        self.TrainFlag = train
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
            Random_flip_layer(probability=0.5),
            Circular_Shift_layer(probability=0.8),
            FRP_RandomCrop(probability = 0.8, factor=self.factor),
            #transforms.ToPILImage(),
            transforms.Resize(image_size),
            AddGaussianNoise(std=self.noise_level),
            transforms.ToTensor(),#normaliza a [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        image_tensor = preprocess(img)
        return image_tensor

    def Transformations_test(self,img):
        preprocess = transforms.Compose([
            transforms.Resize(image_size),
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

class Dataset_FPR_CL(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dir_img, list_IDs, list_labels, unique_labels, multi_instance=False, scale=0.8, noise_level=0.05):
        'Initialization'
        self.dir = dir_img
        self.list_labels = list_labels
        self.list_IDs = list_IDs
        self.unique_labels = unique_labels
        self.multi_instance = multi_instance
        self.factor = scale
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

        preprocess = transforms.Compose([
            Random_flip_layer(probability=0.5),
            Circular_Shift_layer(probability=0.8),
            FRP_RandomCrop(probability = 0.8, factor=self.factor),
            #transforms.ToPILImage(),
            transforms.Resize(image_size),
            AddGaussianNoise(std=self.noise_level),
            transforms.ToTensor(),#normaliza a [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        image_tensor = preprocess(img_adapteq)
        return image_tensor

class FRP_RandomCrop(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, probability = 0.8, factor=0.8):
        self.factor = factor
        self.probability = probability

    def __call__(self, img):
        if np.random.rand() < self.probability:
            image = np.array(img)
            width = image.shape[1]
            new_width = np.rint(width * self.factor)
            reduction = width - new_width
            origin = np.random.randint(low=0,high=reduction, dtype=int)
            image2 = image[origin:new_width.astype(int),origin:new_width.astype(int),:]
            return Image.fromarray(image2.astype('uint8'))
        else:
            return img


    def __repr__(self):
        return self.__class__.__name__+ '(probability={0}, factor={1})'.format(self.probability, self.factor)

class Circular_Shift_layer(object):
  
    def __init__(self, probability = 0.8):
        self.probability = probability

    def __call__(self, img):
        if np.random.rand() < self.probability:
            image = np.array(img)
            width = image.shape[1]
            i0 = np.random.randint(low=1,high=np.rint(width/2), dtype=int)
            for _ in range(i0):
                image = np.roll(image, (-1, -1), axis=(0, 1))

            return Image.fromarray(image.astype('uint8'))
        else:
            return img


    def __repr__(self):
        return self.__class__.__name__+ '(probability={0})'.format(self.probability)



class Random_flip_layer(object):
  
    def __init__(self, probability = 0.8):
        self.probability = probability
        self.RandomVertical = transforms.RandomVerticalFlip(p=1)
        self.RandomHorizontal = transforms.RandomHorizontalFlip(p=1)

    def __call__(self, img):
        if np.random.rand() < self.probability:
            img = self.RandomVertical(img)
            return self.RandomHorizontal(img)
        else:
            return img


    def __repr__(self):
        return self.__class__.__name__+ '(probability={0})'.format(self.probability)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, img):
        image = np.array(img)
        mean = np.mean(image)
        image = image + np.random.randn(*image.shape)* mean*self.std + self.mean
        return Image.fromarray(image.astype('uint8'))
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)