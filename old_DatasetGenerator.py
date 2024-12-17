import os
import numpy as np
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from utils import get_pruned_example_names, get_pruned_example_names_random

#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform,prune_ratio=None,example_not_to_consider = None,train = None, unforgettable_example = None):

        remove_example = None

        if(unforgettable_example!=None):
            file_path = unforgettable_example

        # Read and process the file
            print(f'file path is for example_forgetting - {file_path}')
            with open(file_path, "r") as file:
                data = []
                for line in file:
                    # Replace spaces with commas and split by commas
                    row = line.replace(" ", ",").split(",")
                    data = [value for value in row if value]  # Convert to integers

            remove_example = set(data)
            print(len(data))


        elif(example_not_to_consider=='random_prune'):
            print('dong random pruning with ratio - ', prune_ratio)
            remove_example = get_pruned_example_names_random(example_not_to_consider,pathDatasetFile,pathImageDirectory,prune_ratio)

        elif(example_not_to_consider):
            remove_example = get_pruned_example_names(example_not_to_consider,pathDatasetFile,pathImageDirectory,prune_ratio)

    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
    
        #---- Open file, get image paths and labels
        pruned_sample_count = 0
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        dic = defaultdict(list)
        
        while line:
                
            line = fileDescriptor.readline()
            #--- if not empty
            if line:
          
                lineItems = line.split(",")
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:-1]
                imageLabel = [int(i) for i in imageLabel]
                ind = imageLabel.index(max(imageLabel))
                if(ind not in dic):
                    dic[ind] = [0,0]
                dic[ind][1]+=1
                if((train and example_not_to_consider and imagePath in remove_example) or (unforgettable_example!=None and lineItems[0] in remove_example)):
                    pruned_sample_count+=1
                    dic[ind][0] += 1

                    continue
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)  
            
        fileDescriptor.close()
        li =[[dic[i][0],dic[i][1],i,(dic[i][0]/dic[i][1])] for i in dic.keys()]
        li.sort(key = lambda i: i[0], reverse=True)
        print(li)
        print('pruned ',pruned_sample_count, 'samples')



    
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
    