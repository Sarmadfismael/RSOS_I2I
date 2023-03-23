from torchvision.io import read_image
from torch.utils.data import Dataset
import os

class DataSetRead(Dataset):
    def __init__(self,imgPath,gtPath):
        
        self.imgPath = imgPath
        self.gtPath = gtPath
        self.imageList = os.listdir(imgPath)
        self.length_dataset = len(self.imageList)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
    
        
    
        image = read_image(self.imgPath + '/' + self.imageList[idx]).float()/255
        gt    = read_image(self.gtPath + '/' + self.imageList[idx])[0].long()
        
      
        
        
        
        return image, gt
