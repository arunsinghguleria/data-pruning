import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform,topKClass = 20,filteredLabel = None,type = 'train'):
    
        self.listImagePaths = []
        self.listImageLabels = []
        isFirstIteration = True
        self.numOfLabels = None
        self.labelCount = None
        self.transform = transform
        self.topKClass = topKClass
        self.targets = None
    
        #---- Open file, get image paths and labels
        if(type == 'train'):
            fileDescriptor = open(pathDatasetFile, "r")
            
            line = True
            
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split(",")
                    imageLabel = lineItems[1:-1]
                    imageLabel = [int(i) for i in imageLabel]

                    if(isFirstIteration):
                        isFirstIteration = False
                        self.numOfLabels = len(imageLabel)
                        self.labelCount = [0] * self.numOfLabels

                    for i in range(self.numOfLabels):
                        self.labelCount[i]+=imageLabel[i]

            fileDescriptor.close()

            self.labelCount = [[self.labelCount[i],i] for i in range(self.numOfLabels)]
            self.labelCount.sort(key= lambda i: i[0],reverse=True)
            self.labelCount = self.labelCount[:self.topKClass]
            
            self.filteredLabel = [i[1] for i in self.labelCount]
            self.targets = self.filteredLabel
            
            # print(pathDatasetFile)
            print('label count -',self.labelCount)
            print('label filtered -',self.filteredLabel)

        else:
            self.filteredLabel = filteredLabel
    
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
                
                one_index = imageLabel.index(max(imageLabel))
                
                if(one_index in self.filteredLabel):
                    
                    self.listImagePaths.append(imagePath)
                    self.listImageLabels.append(imageLabel)  
            
        fileDescriptor.close()

        # self.labelCount = [[self.labelCount[i],i] for i in range(self.numOfLabels)]
        # self.labelCount.sort(key= lambda i: i[0],reverse=True)



        print('length of dataset - ',len(self.listImagePaths))
        # print(len(self.listImageLabels[0]))
        
        self.n = len(self.listImagePaths)
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        
        if self.transform != None: imageData = self.transform(imageData)
        # return imageData, torch.argmax(imageLabel)
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        # return len(self.listImagePaths)
        return self.n
    

    def get_data(self,index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        
        if self.transform != None: imageData = self.transform(imageData)
        # return imageData, torch.argmax(imageLabel)
        return imageData, imageLabel, imagePath
    
 #-------------------------------------------------------------------------------- 

    