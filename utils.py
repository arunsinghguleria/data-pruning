import torch
import torch.nn as nn
from torchvision import  transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights, Inception_V3_Weights
from torchvision.transforms.functional import crop
from tqdm import tqdm
from collections import defaultdict
import os

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


def calculate_GraNd_score(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def calculate_EL2N_score(logit,labels):
    logit = torch.softmax(logit,dim=1)

    logit -= labels
    logit = logit ** 2
    logit = logit.sum().item()
    logit = logit ** 0.5
    return logit





def get_scores(model,m4_train_data,optimizer,criterion,device,df_EL2N_score,df_GraNd_score,epoch_num):

    '''
        function will take model and train_sample as input, do inference, calculate GraNd score and EL2N score and will sotre the result in a file.
    '''

    print('---> in get score method <---')
    col_name = 'epoch_'+str(epoch_num)
    df_EL2N_score[col_name] = 0
    df_GraNd_score[col_name] = 0
    print(df_EL2N_score)
    print(df_GraNd_score)

    model.eval()
    # for batch_no, (images, labels) in enumerate(tqdm(nih_dataLoaderTrain_batch_1)):
    for i in tqdm(range(m4_train_data.n)):
        images,labels, image_path = m4_train_data.get_data(i)
        # print(images.shape)
        images = images.unsqueeze(0)
        labels = labels.unsqueeze(0)
        # print(images.shape)
        images = images.to(device)
        labels = labels.to(device)
        # zeroing the optimizer
        optimizer.zero_grad()
            
        outputs = model(images)
        _, labels_index = labels.max(1)
        loss = criterion(outputs, labels_index)   
        loss.backward()

        GraNd_score = calculate_GraNd_score(model)
        df_GraNd_score.loc[image_path,col_name] = float(GraNd_score)

        EL2N_score = calculate_EL2N_score(outputs,labels)
        df_EL2N_score.loc[image_path,col_name] = float(EL2N_score)        
    
    df_EL2N_score.to_csv(f'./Scores/EL2N_score_{epoch_num}.csv') 
    df_GraNd_score.to_csv(f'./Scores/GraNd_score_{epoch_num}.csv')






def get_remove_example_names(path,pathDatasetFile,pathImageDirectory,prune_ratio ,epoch_no = 6,ratio = 0.2):
    '''
        function will take GraNd score file or EL2N score file as input, and return the names of smaples which are to pruned in DatasetGenerter.
    '''
    listImageLabels = defaultdict(list)

    dataset_dict = {}
    
    fileDescriptor = open(pathDatasetFile, "r")
        
    #---- get into the loop
    line = True
    
    while line:
                
        line = fileDescriptor.readline()
            
            #--- if not empty
        if line:
          
            lineItems = line.split(",")
                
            imagePath = os.path.join(pathImageDirectory, lineItems[0])
            imageLabel = lineItems[1:-1]
            imageLabel = [int(i) for i in imageLabel]
            imageLabel = imageLabel.index(max(imageLabel))
            dataset_dict[imagePath] = imageLabel
    fileDescriptor.close()

    
    fileDescriptor = open(path, "r")
        
        #---- get into the loop
    line = fileDescriptor.readline()

    line = fileDescriptor.readline() # added to ignore the first line (column names)

    cnt = 0
        
    while line:
                
        if line:
          
            lineItems = line.split(",")

            img_name = [lineItems[0]]
            
            score = [ float(i) for i in lineItems[1:] ]

            listImageLabels[dataset_dict[img_name[0]]].append(img_name+score)
            
        line = fileDescriptor.readline()

    fileDescriptor.close()
    
    li = []
    for k in listImageLabels.keys():
        listImageLabels[k] = sorted(listImageLabels[k],key = lambda i: i[-1])
        li.append([k,len(listImageLabels[k])])
    
    li = sorted(li,key = lambda i: i[1],reverse = True)
    
    for i in range(len(li)):
        k = li[i][0]
        cnt = li[i][1]
        ratio = prune_ratio[i]
    
        listImageLabels[k] = listImageLabels[k][:int(ratio*cnt)]
    
    pruned_image_names = set()

    for k in listImageLabels.keys():
        for image in listImageLabels[k]:
            pruned_image_names.add(image[0])

    return pruned_image_names