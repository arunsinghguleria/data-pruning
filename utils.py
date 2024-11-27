import torch
import torch.nn as nn
from torchvision import  transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights, Inception_V3_Weights
from torchvision.transforms.functional import crop

import logging

def create_logger(name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

class DenseNet121_Multi_Class(nn.Module):
    """Model for training densenet baseline"""
    def __init__(self, classCount, isTrained=False):
        super(DenseNet121_Multi_Class, self).__init__()
        # self.densenet = models.densenet121(weights = DenseNet121_Weights.DEFAULT)
        self.densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', weights = 'DenseNet121_Weights.IMAGENET1K_V1')
        self.features = self.densenet.features
        self.kernelCount = self.densenet.classifier.in_features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1,1)) 
        self.classifier = nn.Sequential(nn.Linear(self.kernelCount, classCount))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # flatten the x
        x = x.view((x.shape[0],-1))
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x

class Inception_Multi_Class(nn.Module):
    """Model for training densenet baseline"""
    def __init__(self, classCount, isTrained=False):
        super(Inception_Multi_Class, self).__init__()
        self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='Inception_V3_Weights.IMAGENET1K_V1')
        self.kernelCount = self.inception.fc.in_features
        self.inception.fc = nn.Linear(self.kernelCount, classCount)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.inception(x)
        
        if hasattr(x, 'logits'):
            x = self.sigmoid(x.logits)
        else:
            x = self.sigmoid(x)    
        return x


class ResNet_Multi_Class(nn.Module):
    """Model for training densenet baseline"""
    def __init__(self, classCount, isTrained=False):
        super(ResNet_Multi_Class, self).__init__()
        # self.densenet = models.densenet121(weights = DenseNet121_Weights.DEFAULT)
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.IMAGENET1K_V1')
        # print(self.resnet)
        self.kernelCount = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.kernelCount, classCount)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x     



class ResNeXt_Multi_Class(nn.Module):
    """Model for training densenet baseline"""
    def __init__(self, classCount, isTrained=False):
        super(ResNeXt_Multi_Class, self).__init__()
        self.resnext = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2')
        # print(self.resnet)
        self.kernelCount = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(self.kernelCount, classCount)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.resnext(x)
        # x = self.softmax(x)
        return x                



def croptop(image):
    #print("Image size", image.size)
    width, height = image.size
    return crop(image, int(.08*height), 0, height, width)

def get_dataloader_preprocess(path: str, batch_size: int, image_size:int, num_workers:int):
    """"Image Dataloader that returns a path"""
    transform = transforms.Compose([
                    transforms.Lambda(croptop),
                    transforms.CenterCrop(image_size),
                    transforms.RandomApply(torch.nn.ModuleList([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(degrees=0,translate=(.1,.1)),
                    transforms.ColorJitter(brightness=(.9,1.1)),
                    transforms.RandomAffine(degrees=0,scale=(0.85, 1.15)),
                    ]), p=0.5),
                    #transforms.Normalize(mean=[ 0.406], std=[0.225]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    dataset = ImageFolder(path, transform = transform)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle = True, num_workers=num_workers, drop_last=False)

    return dataloader, len(dataset)    


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index) -> tuple:
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return image, label, path
    
def get_dataloader_paths(path: str, batch_size: int, image_size:int, num_workers:int):
    """"Image Dataloader that returns a path"""
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])])
    dataset = ImageFolderWithPaths(path, transform = transform)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle = False, num_workers=num_workers, drop_last=False)

    return dataloader, len(dataset)


def calculate_classwise_accuracy(li):
    tot = 0
    i = 0
    head = 0
    med = 0
    tail = 0
    while(i<7):
        head+=li[i][1]
        tot+=li[i][1]
        i+=1
    head/=7
    while(i<16):
        med+=li[i][1]
        tot+=li[i][1]
        i+=1
    med/=9

    while(i<20):
        tail+=li[i][1]
        tot+=li[i][1]
        i+=1
    tail/=4
    tot/=20

    return {'total': tot,
            'head': head,
            'medium' :med,
            'tail': tail}