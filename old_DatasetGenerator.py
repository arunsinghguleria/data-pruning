import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from utils import get_remove_example_names

#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform,prune_ratio=None,example_not_to_consider = None,train = None):

        remove_example = None

        if(example_not_to_consider):
            remove_example = get_remove_example_names(example_not_to_consider,pathDatasetFile,pathImageDirectory,prune_ratio)

    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split(",")
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                if(train and example_not_to_consider and imagePath in remove_example):
                    continue
                imageLabel = lineItems[1:-1]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)  
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        
        if self.transform != None: imageData = self.transform(imageData)
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 
    