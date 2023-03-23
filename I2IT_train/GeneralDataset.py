from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision





class PVDataSet(Dataset):
    def __init__(self, root_P,root_V):  
        
        self.root_P = root_P
        self.root_V = root_V
       
        

        
        self.V_images = os.listdir(root_V)
        self.P_images = os.listdir(root_P)
        self.length_dataset = max(len(self.V_images), len(self.P_images)) 
        self.V_len = len(self.V_images)
        self.P_len = len(self.P_images)



    def __len__(self):
        return self.length_dataset


    def __getitem__(self, index):
        V_img = self.V_images[index % self.V_len]
        P_img = self.P_images[index % self.P_len]

        V_path = os.path.join(self.root_V, V_img)
        P_path = os.path.join(self.root_P, P_img)
        V_img = np.array(Image.open(V_path).convert("RGB"))
        P_img = np.array(Image.open(P_path).convert("RGB"))


        V_img = np.asarray(V_img, np.float32)
        V_img = torch.from_numpy(V_img)
        V_img = V_img.permute(2, 0, 1)
        V_img = V_img.float()

        P_img = np.asarray(P_img, np.float32)
        P_img = torch.from_numpy(P_img)
        P_img = P_img.permute(2, 0, 1)
        P_img = P_img.float()
        
        return P_img,V_img