import os
from glob import glob
import numpy as np
from tqdm import tqdm
from skimage.io import imread,imsave 
import matplotlib

from matplotlib.image import imread 
#from skimage.color import grey2rgb 
import matplotlib.pyplot as plt
import random, numbers

import torch
import torch.utils.data as data 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from PIL import Image
import Augmentor
import SimpleITK as sitk
from skimage import transform
from skimage.measure import regionprops
import pdb

__all__ = ['ElasticTransform', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'ToTensor', 'Lung_dataset', 'Normalize']

# class ElasticTransform(object):
#     def __init__(self, mode='train'):
#         self.mode = mode

#     def __call__(self, sample):
#         #print(self.mode)
#         if self.mode == 'train':
#             image, target = sample['image'], sample['target']
#             images = [[image, target]]

#             p = Augmentor.DataPipeline(images)
#             # resize
#             #p.resize(probability=1, width=512, height=512)
#             # random flip  
#             p.flip_left_right(probability=0.5)
#             # random elastic distortation
#             #grid_width = np.random.randint(5, 10)
#             #grid_height = np.random.randint(5, 10)
#             #mag = np.random.randint(2, 8)
#             #p.random_distortion(0.4, grid_width, grid_height, mag)
#             sample_aug = p.sample(1)

#             # sample['image'] = grey2rgb(sample_aug[0][0])
#             # sample['target'] = grey2rgb(sample_aug[0][1])
#             sample['image'] = sample_aug[0][0]                   # for dicom
#             sample['target'] = sample_aug[0][1]                  # for dicom

#             return sample

#         if self.mode == 'test':
#             image, target = sample['image'], sample['target']
#             images = [[image, target]]

#             p = Augmentor.DataPipeline(images)
#             # resize
#             p.resize(probability=1, width=512, height=512)

#             sample_aug = p.sample(1)
#             sample['image'] = grey2rgb(sample_aug[0][0])
#             sample['target'] = grey2rgb(sample_aug[0][1])
#             return sample


class RandomHorizontalFlip(object):
    def __init__(self, p=0.6):
        self.p = p

    def __call__(self, sample):
        image, image2, target = sample['image'], sample['image_b'], sample['target']
        
        #pdb.set_trace()
        if random.random() < self.p:
            sample['image'] = F.hflip(image)
            sample['image_b'] = F.hflip(image2)
            sample['target'] = F.hflip(target)
            
            return sample
        return sample


class RandomVerticalFlip(object):
    def __init__(self, p=0.6):
        self.p = p

    def __call__(self, sample):
        image, image2, target = sample['image'], sample['image_b'], sample['target']

        if random.random() < self.p:
            sample['image'] = F.vflip(image)
            sample['image_b'] = F.vflip(image2)
            sample['target'] = F.vflip(target)
            
            return sample
        return sample


class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
    
    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, sample):
        image, image2, target = sample['image'], sample['image_b'], sample['target']
        angle = self.get_params(self.degrees)
        sample['image'] = F.rotate(image, angle, self.resample, self.expand, self.center)
        sample['image_b'] = F.rotate(image2, angle, self.resample, self.expand, self.center)
        sample['target'] = F.rotate(target, angle, self.resample, self.expand, self.center)

        return sample

class Crop(object):
    def __init__(self, mode='train'):
        self.mode = mode
    
    def __call__(self, sample):
        image, image2, target = sample['image'], sample['image_b'], sample['target']
        # pdb.set_trace()
        image = np.asarray(image)
        image2 = np.asarray(image2)
        target = np.asarray(target)
        size = image.shape[0]
        target = target.astype('int16')
        props = regionprops(target)
        centroid = props[0].centroid
        centroid = np.ceil(centroid).astype(np.int16)
        bboxtuple = props[0].bbox
        x1,y1,x2,y2 = bboxtuple[1], bboxtuple[0], bboxtuple[3], bboxtuple[2]
        
        if x1 >= 128 and x2 <= size - 128 and y1 >= 128 and y2 <= size - 128:
            cropimg = image[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
            cropimg2 = image2[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
            croptarget = target[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
        elif y2 > size - 128:
            centroid[0] = centroid[0] - (y2 + 128 - size)
            cropimg = image[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
            cropimg2 = image2[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
            croptarget = target[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
        elif x2 > size - 128:
            centroid[1] = centroid[1] - (x2 + 128 - size)    
            cropimg = image[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
            cropimg2 = image2[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
            croptarget = target[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
        elif x1 < 128:
            centroid[1] = centroid[1] + (128 - x1)
            cropimg = image[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
            cropimg2 = image2[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
            croptarget = target[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
        elif y1 < 128:
            centroid[0] = centroid[0] + (128 - y1)
            cropimg = image[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
            cropimg2 = image2[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]
            croptarget = target[centroid[0]-128 : centroid[0]+128, centroid[1]-128 : centroid[1]+128]

        
        # if x1 >= 96 and x2 <= size - 96 and y1 >= 96 and y2 <= size - 96:
        #     cropimg = image[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        #     cropimg2 = image2[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        #     croptarget = target[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        # elif y2 > size - 96:
        #     centroid[0] = centroid[0] - (y2 + 96 - size)
        #     cropimg = image[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        #     cropimg2 = image2[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        #     croptarget = target[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        # elif x2 > size - 96:
        #     centroid[1] = centroid[1] - (x2 + 96 - size)    
        #     cropimg = image[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        #     cropimg2 = image2[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        #     croptarget = target[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        # elif x1 < 96:
        #     centroid[1] = centroid[1] + (96 - x1)
        #     cropimg = image[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        #     cropimg2 = image2[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        #     croptarget = target[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        # elif y1 < 96:
        #     centroid[0] = centroid[0] + (96 - y1)
        #     cropimg = image[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        #     cropimg2 = image2[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]
        #     croptarget = target[centroid[0]-96 : centroid[0]+96, centroid[1]-96 : centroid[1]+96]


        cropImg = Image.fromarray(cropimg)
        cropImg2 = Image.fromarray(cropimg2)
        cropTarget = Image.fromarray(croptarget)
        
        return {'image': cropImg, 'image_b': cropImg2, 'target': cropTarget}


class ToTensor(object):
    def __init__(self, mode='train'):
        self.mode = mode
    
    def __call__(self, sample):
        # if self.mode == 'train':
            # pdb.set_trace()
        image, image2, target = sample['image'], sample['image_b'], sample['target']
        # image = image.astype(np.uint8)
        # image2 = image2.astype(np.uint8)
        #image = np.expand_dims(image[:, :, 0], 0)
        # target = np.expand_dims(target, 0)
        # image = image.transpose(2, 0, 1) 
        image = np.asarray(image)[np.newaxis, :,:]
        image2 = np.asarray(image2)[np.newaxis, :,:]
        target = np.asarray(target)[np.newaxis, :,:]
        
        image = torch.from_numpy(image)
        image2 = torch.from_numpy(image2)
        # image = image.float().div(1024)
        # transverse tensor to 0~1 
        # if isinstance(image, torch.ByteTensor): 
        #     image = image.float().div(255)
        #     image2 = image2.float().div(255)
        # pdb.set_trace()
        image  = image.float().div(4095)        # T2
        image2 = image2.float().div(4095)       # DWI
        image[image>1] = 1
        image2[image2>1] = 1

        return {'image':image, 'image_b':image2, 'target':torch.from_numpy(target.astype(np.float32))}
        

class Normalize(object):
    def __init__(self, mean, std, mode='train'):
        self.mode = mode 
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        # if self.mode == 'train':
        image, image2, target = sample['image'], sample['image_b'], sample['target']
        image  = (image - self.mean) / self.std
        image2 = (image2 - self.mean) / self.std

        sample['image'] = image
        sample['image_b'] = image2
        return sample


class Lung_dataset(data.Dataset):
    ''' ABUS_Dataset class, return 2d transverse images and targets ''' 
    def __init__(self, image_path=None, image2_path=None, target_path=None, transform=None, mode='train'): 
        if image_path is None: 
            raise(RuntimeError("image_path must be set"))
        if target_path is None and mode != 'infer':
            raise(RuntimeError("both image_path and target_path must be set if mode is not 'infer'"))
        data_file_names, data2_file_names, label_file_names, target_means = Lung_dataset.get_all_filenames(image_path, image2_path, target_path)
        if len(data_file_names) == 0:
            raise(RuntimeError("Found 0 images in : " + os.path.join(image_path) + "\n"))

        self.data_file_names = data_file_names
        self.data2_file_names = data2_file_names
        self.label_file_names = label_file_names
        self._target_means = target_means
        self.mode = mode
        self.image_path = image_path
        self.image2_path = image2_path
        self.target_path = target_path
        self.transform = transform

    def __getitem__(self, index):
        # self.data_file_names.sort()
        # self.data2_file_names.sort()
        # self.label_file_names.sort()
        data_file_name = self.data_file_names[index]
        data2_file_name = self.data2_file_names[index]
        label_file_name = self.label_file_names[index]
        
        #pdb.set_trace()
        sitktarget, target = Lung_dataset.load_image(self.target_path, label_file_name)

        # load image
        sitkimg, image = Lung_dataset.load_image(self.image_path, data_file_name)
        sitkimg2, image2 = Lung_dataset.load_image(self.image2_path, data2_file_name)
        target = target.astype('float32')
        image = np.squeeze(image)
        image2 = np.squeeze(image2)
        target = np.squeeze(target)

        image = Image.fromarray(image)
        image2 = Image.fromarray(image2)
        target = Image.fromarray(target)

        sample = {'image':image, 'image_b':image2, 'target':target}
        #pdb.set_trace()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data_file_names)

    def get_target_mean(self):
        return self._target_means

    @staticmethod
    def get_all_filenames(image_path, image2_path, target_path):
        '''
        get all filenames in target_path
        
        ---
        return:

        all_filenames: all filenames
        
        target_means: used for weighted cross entropy loss

        '''
        list1 = os.listdir(image_path)
        list2 = os.listdir(target_path)
        list3 = os.listdir(image2_path)
        list1.sort()
        list2.sort()
        list3.sort()
        # pdb.set_trace()
        #all_filenames = glob(os.path.join(target_path, '*.png'))
        # all_filenames = [file_name for file_name in os.listdir(target_path) if file_name.endswith('png')]
        all_data_filenames  = [file_name for file_name in list1 if file_name.endswith('dcm')]
        all_label_filenames = [file_name for file_name in list2  if file_name.endswith('dcm')]
        all_data2_filenames = [file_name for file_name in list3  if file_name.endswith('dcm')]
        #print(all_filenames) 
        # count target mean in train_set 
        target_mean = [] 
        for file_name in all_label_filenames:
            sitktarget, target = Lung_dataset.load_image(target_path, file_name)
            temp_mean = np.mean(target)
            if temp_mean != 0:
                target_mean.append(temp_mean)
        target_mean = np.mean(target_mean)

        return all_data_filenames, all_data2_filenames, all_label_filenames, target_mean
        
    @staticmethod
    def load_image(file_path, file_name):
        full_name = os.path.join(file_path, file_name)
        #img = imread(full_name) 
        # print('full_name: ', full_name)
        img = sitk.ReadImage(full_name)
        img_array = sitk.GetArrayFromImage(img)      # sitk (z,y,x)

        # we don't normalize image when loading them, because Augmentor will raise error
        # if nornalize, normalize origin image to mean=0,std=1.
        #if is_normalize:
        #    img = img.astype(np.float32)
        #    mean = np.mean(img)
        #    std = np.std(img)
        #    img = (img - mean) / std

        return img, img_array 


# if __name__ == '__main__':
#     # test bjtu_dataset_2d
#     image_path = '../isia_dataset_2d/test_data/'
#     target_path = '../isia_dataset_2d/test_label/'
    
#     transform = transforms.Compose([ToTensor(),
#                                     Normalize(0.5, 0.5)
#                                     ])

#     train_set = Lung_dataset(image_path, target_path, transform)
#     train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
#     for sample in train_loader:
#         image, target = sample['image'], sample['target']
#         print('image shape: ', image.shape)
#         print('label shape: ', target.shape)
