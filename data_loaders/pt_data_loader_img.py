 #This library contains classes and functions to load FRPs and apply custom data 
 #augmentation operations to models implemented in PyTorch.

import torch
from torchvision import transforms
import numpy as np
import os
import cv2
from PIL import Image, ImageOps


image_size = 224

class Dataset_Spec_img(torch.utils.data.Dataset):
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
            transforms.Resize((image_size,image_size)),
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
            transforms.Resize((image_size,image_size)),
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

class Dataset_Spec_img_CL(torch.utils.data.Dataset):
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
            transforms.Resize((image_size,image_size)),
            AddGaussianNoise(std=self.noise_level),
            transforms.ToTensor(),#normaliza a [0,1]
            Time_Masking_layer(p=0.7, scale=self.scale),
            Freq_Masking_layer(p=0.7, scale=self.scale),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        image_tensor = preprocess(img_adapteq)
        return image_tensor


class Dataset_Spec_txtmat(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dir_img, list_IDs, list_labels, unique_labels, train=True, multi_instance=False, scale=(0.05, 0.1), noise_level=0.05):
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
            transforms.Resize((image_size,image_size)),
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
            transforms.Resize((image_size,image_size)),
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
            img = np.loadtxt(img_path + file_exp[0] + '_' + n_sp + os.sep + 'Axis_' + axis + os.sep + file_exp[4] + '.txt' , dtype=float, delimiter=',')
        else:
            img = np.loadtxt(img_path + ID +'.txt', dtype=float, delimiter=',')
        #---------------------------------------------------------------
        image = img*255
        image2 = np.copy(image)
        image2[image2>0]=255
        mask = Image.fromarray(image2.astype('uint8'))
        #Create 3 channels
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        #---------------------------------------------
        img = Image.fromarray(image.astype('uint8'), 'RGB')
        img_adapteq = ImageOps.equalize(img,mask=mask)

        #---------------------------------------------------------------
        if self.TrainFlag:
            return self.Transformations_train(img_adapteq)
        else:
            return self.Transformations_test(img_adapteq)

class Dataset_Spec_txtmat_CL(torch.utils.data.Dataset):
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
    
    def load_image(self, img_path,ID):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        if self.multi_instance:
            file_exp = ID.split('_')
            n_sp = file_exp[1]
            axis = file_exp[3]
            img = np.loadtxt(img_path + file_exp[0] + '_' + n_sp + os.sep + 'Axis_' + axis + os.sep + file_exp[4] + '.txt' , dtype=float, delimiter=',')
        else:
            img = np.loadtxt(img_path + ID +'.txt', dtype=float, delimiter=',')

        #---------------------------------------------------------------
        image = img*255
        image2 = np.copy(image)
        image2[image2>0]=255
        mask = Image.fromarray(image2.astype('uint8'))

        #Create 3 channels
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        #---------------------------------------------
        img = Image.fromarray(image.astype('uint8'), 'RGB')
        img_adapteq = ImageOps.equalize(img,mask=mask)

        #---------------------------------------------------------------
        preprocess = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.ToPILImage(),
            transforms.Resize((image_size,image_size)),
            AddGaussianNoise(std=self.noise_level),
            transforms.ToTensor(),#normaliza a [0,1]
            Time_Masking_layer(p=0.7, scale=self.scale),
            Freq_Masking_layer(p=0.7, scale=self.scale),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        image_tensor = preprocess(img_adapteq)
        return image_tensor.type(torch.FloatTensor)

class Time_Masking_layer(object):
  
    def __init__(self, p = 0.8, scale=(0.1,0.2)):
        self.probability = p
        self.scale = scale

    def __call__(self, img):
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        factor = (self.scale[0] - self.scale[1]) * torch.rand(1) + self.scale[1]
        w = np.int(img_w*factor)
        j = torch.randint(0, img_w - w + 1, size=(1,)).item()
        v =torch.empty([img_c, img_h, w], dtype=torch.float32).normal_()
        v -= v.min(1, keepdim=True)[0]
        v /= v.max(1, keepdim=True)[0]
        #v =torch.zeros([img_c, img_h, w], dtype=torch.float32)
        if np.random.rand() < self.probability:
            return transforms.functional.erase(img,0,j,img_h,w,v)
        else:
            return img
    
    def __repr__(self):
        return self.__class__.__name__+ '(probability={0}, scale={1})'.format(self.probability, self.scale)

class Freq_Masking_layer(object):
  
    def __init__(self, p = 0.8, scale=(0.1,0.2)):
        self.probability = p
        self.scale = scale

    def __call__(self, img):
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        factor = (self.scale[0] - self.scale[1]) * torch.rand(1) + self.scale[1]
        h = np.int(img_h*factor)
        i = torch.randint(0, img_h - h + 1, size=(1,)).item()
        v =torch.empty([img_c, h, img_w], dtype=torch.float32).normal_()
        v -= v.min(1, keepdim=True)[0]
        v /= v.max(1, keepdim=True)[0]
        #v =torch.zeros([img_c, h, img_w], dtype=torch.float32)
        if np.random.rand() < self.probability:
            return transforms.functional.erase(img,i,0,h,img_w,v)
        else:
            return img
    
    def __repr__(self):
        return self.__class__.__name__+ '(probability={0}, scale={1})'.format(self.probability, self.scale)

class Random_flip_layer(object):
  
    def __init__(self, p = 0.8, imgf=True):
        self.probability = p
        self.RandomHorizontal = transforms.RandomHorizontalFlip(p=1)
        self.imgf = imgf

    def __call__(self, img):
        if self.imgf:
            if np.random.rand() < self.probability:
                return self.RandomHorizontal(img)
            else:
                return img
        else:
            if np.random.rand() < self.probability:
                return np.flip(img,1).copy()
            else:
                return img


    def __repr__(self):
        return self.__class__.__name__+ '(probability={0})'.format(self.probability)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., imgf=True):
        self.std = std
        self.mean = mean
        self.imgf = imgf
        
    def __call__(self, img):
        image = np.array(img)
        mean = np.mean(image)
        image = image + np.random.randn(*image.shape)* mean*self.std + self.mean
        if self.imgf:
            image = np.clip(image, 0, 255)
            return Image.fromarray(image.astype('uint8'))
        else:
            image = np.clip(image, 0, 1)
            return image
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
