import numpy as np
import pandas as pd
import argparse
import itertools
import time 
from tqdm import tqdm

from torchvision import transforms, models, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt


from GeneralDataset import PVDataSet
from models import Encoder, Decoder , Perceptual,Discriminator,VGG19_FeaturesExtractor
import utils




total_epochs = 1

max_lr = 1e-3
input_nc = 3
output_nc = 3
Hsize = 512 
Wsize = 512 
batchSize = 1




RES_DIR = 'Result'
OUTPUT_DIR = "Output_Images"

# Create output dirs if they don't exist
if not os.path.exists(RES_DIR + '/' + OUTPUT_DIR):
    os.makedirs(RES_DIR + '/' + OUTPUT_DIR)





# Check if GPU is available 
avail = torch.cuda.is_available()
device = torch.device("cuda" if avail else "cpu")
devCnt = torch.cuda.device_count()
devName = torch.cuda.get_device_name(0)
print("Available: " + str(avail) + ", Count: " +
      str(devCnt) + ", Name: " + str(devName))




#read the target image dir 
SourceTrainImageDataset = "path"
SourceValImageDataset   = "path"
SourceTestImageDataset  = "path"

#read the source image dir 
targetTrainImageDataset = "path"
targetValImageDataset   = "path"
targetTestImageDataset  = "path"


# Create the dataloader  
trainDS = PVDataSet(SourceTrainImageDataset,targetTrainImageDataset)
trainDL = DataLoader(trainDS, batch_size=batchSize, shuffle=True, sampler=None,
                                      batch_sampler=None, num_workers=0, collate_fn=None,
                                      pin_memory=False, drop_last=False , timeout=0,
                                     worker_init_fn=None)

# Create the datasets  
valDS = PVDataSet(SourceValImageDataset,targetValImageDataset)
valDL = DataLoader(valDS, batch_size=batchSize, shuffle=True, sampler=None,
                                      batch_sampler=None, num_workers=0, collate_fn=None,
                                      pin_memory=False, drop_last=False , timeout=0,
                                      worker_init_fn=None)
 

# read the source file names 
Source_trainFileName = np.loadtxt("SourceTrain.txt", dtype= 'O')
Source_valFileName   = np.loadtxt("SourceVal.txt", dtype= 'O')


# define the model 
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()
perceptual = Perceptual(encoder, decoder)
VGG19_features = VGG19_FeaturesExtractor()

# send to device 
perceptual.to(device)
discriminator.to(device)
VGG19_features.to(device);

# defince the losses
criterion_identity = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()

# define the optimizer and lr_scheduler
optimizer_Perceptual = torch.optim.Adam(perceptual.parameters(),lr=max_lr, betas=(0.5, 0.999))
optimizer_D          = torch.optim.Adam(discriminator.parameters(), lr=max_lr, betas=(0.5, 0.999))

Lr_scheduler_Per= torch.optim.lr_scheduler.OneCycleLR(optimizer_Perceptual, max_lr, epochs=total_epochs,steps_per_epoch=len(trainDL))
Lr_scheduler_Dis= torch.optim.lr_scheduler.OneCycleLR(optimizer_D, max_lr, epochs=total_epochs,steps_per_epoch=len(trainDL))



# just to get the Lr value 
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



# read one-shot target image 
real_B = np.array(Image.open('V_Style_Image.png').convert("RGB"))
real_B = np.asarray(real_B, np.float32)
real_B = torch.from_numpy(real_B)
real_B = real_B.permute(2, 0, 1)
real_B = real_B.float()
real_B = real_B.repeat(batchSize, 1, 1, 1)
real_B = real_B.to(device)

# get the VGG contnt and style  
features_style = VGG19_features(real_B)
gram_style = [utils.gram_matrix(y) for y in features_style]



# training part 
reconstructionA_losses_List = []
reconstructionB_losses_List = []

content_loss_List = []
style_loss_List = []
loss_adv_List = []

Total_losses_List = []

time_List = [] 

for epoch in range(total_epochs):
    
    since = time.time()
    
    for i,(real_A,_) in enumerate(tqdm(trainDL, desc='Epoch: {}/{}'.format(epoch, total_epochs))):
        
        iter_start_time = time.time()
        
        real_A = real_A.to(device)

        # update the perceptual 
        
        optimizer_Perceptual.zero_grad()
        
        # get the images  
        mixed_image, reconstructionA, reconstructionB = perceptual(real_A, real_B)
        
        
        real_Ac = Variable(real_A.data.clone(), requires_grad=True)
        
        # Reconstruction loss
        loss_reconstruction_A = criterion_identity(reconstructionA, real_Ac)
        loss_reconstruction_B = criterion_identity(reconstructionB, real_B)
        
        # adv loss
        pred_mixed = discriminator(mixed_image)
        loss_adv   = mse_loss(pred_mixed , torch.ones_like(pred_mixed))
        
        
        features_y  = VGG19_features(mixed_image)
        features_xc = VGG19_features(real_Ac)
        
        
        # content loss
        f_xc_c       = Variable(features_xc[1].data, requires_grad=False)
        content_loss = mse_loss(features_y[1], f_xc_c)

        
        style_loss = 0.
        for m in range(len(features_y)):
                gram_s = Variable(gram_style[m].data, requires_grad=False)
                gram_y = utils.gram_matrix(features_y[m])
                style_loss += 5 * mse_loss(gram_y, gram_s[:, :, :])
        
        #total loss 
        total_loss =30.0 * (loss_reconstruction_A + loss_reconstruction_B) + content_loss + style_loss + 1e3 * loss_adv

        total_loss.backward()
        optimizer_Perceptual.step()
        
        
        # train the Disc 
        optimizer_D.zero_grad()
        
        pred_real = discriminator(real_B)
        loss_D_real = mse_loss(pred_real ,torch.ones_like(pred_real))
        
        pred_fake = discriminator(mixed_image.detach())
        loss_D_fake = mse_loss(pred_fake ,torch.zeros_like(pred_fake))
        
        total_discriminator_loss = 1e3 * (loss_D_real + loss_D_fake)
        
        total_discriminator_loss.backward()
        optimizer_D.step()
        
        t_comp = (time.time() - iter_start_time)       
        
        # Compute the training time 
        time_List.append(t_comp)
        
    reconstructionA_losses_List.append(30.0 * loss_reconstruction_A.item())
    reconstructionB_losses_List.append(30.0 * loss_reconstruction_B.item())
    

    content_loss_List.append(content_loss.item())
    style_loss_List.append(style_loss.item())
    loss_adv_List.append(loss_adv.item())
    Total_losses_List.append(total_loss.item())

    

    
    print("total_loss: {:.3f}".format(total_loss.item()),
      "loss_reconstruction_A: {:.3f}".format(30.0 * loss_reconstruction_A.item()),
      "loss_reconstruction_B: {:.3f}".format(30.0 * loss_reconstruction_B.item()),
      "discriminator: {:.3f}".format(1e3 * loss_adv.item()),
       "total_discriminator_loss: {:.3f}".format(total_discriminator_loss.item()),
      "content_loss: {:.3f}".format(content_loss.item()),
      "style_loss: {:.3f}".format(style_loss.item()),
       "Time: {:.2f}".format(np.mean(time_List)))  




# compute the no. of parameters in MB 
perceptual.load_state_dict(torch.load("./Result/model.pth"))
sum(p.numel() for p in perceptual.parameters() if p.requires_grad)/1000000



