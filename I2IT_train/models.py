import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
from torch.nn.utils import spectral_norm
from batchinstancenorm import BatchInstanceNorm2d as Normlayer
import functools
from functools import partial


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                bin(filters)
            )

    def forward(self, x):
        output = self.main(x)
        output += self.shortcut(x)
        return output
    
    
class Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Encoder, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
            bin(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, x):
        output = self.model(x)
        return output



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            spectral_norm(nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True)),
            # nn.Tanh()
        )

    def forward(self,x ):
        return self.model(x)

class Perceptual(nn.Module):
    def __init__(self, encoder, decoder): #, generator):
        super(Perceptual, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        # self.generator = generator

    def forward(self, A, B):

        # reconstructedA, reconstructedB = self.generator(A, B)

        latentA = self.encoder(A)
        latentB = self.encoder(B)
        
        reconstructedA = self.decoder(latentA)
        reconstructedB = self.decoder(latentB)
        
        latentA.detach()
        latentB.detach()

        
        latentALower= latentA[:, 0:32, :, :]
        latentBUpper  = latentB[:, 32:64, : , :]
        mixed_latent = torch.cat([latentALower,latentBUpper ], dim=1)
        mixed_image = self.decoder(mixed_latent)

        return mixed_image, reconstructedA, reconstructedB
        

        
        
class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class VGG19_FeaturesExtractor(nn.Module):
    def __init__(self):
        super(VGG19_FeaturesExtractor, self).__init__()
        
        features = models.vgg19(pretrained=True).features
        
        self.to_relu_1_1 = nn.Sequential()
        self.to_relu_1_2 = nn.Sequential()
        
        self.to_relu_2_1 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_3_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_3_4 = nn.Sequential()
        
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_4_2 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.to_relu_4_4 = nn.Sequential()
        
        
        self.to_relu_5_1 = nn.Sequential()
        self.to_relu_5_2 = nn.Sequential()
        self.to_relu_5_3 = nn.Sequential()
        self.to_relu_5_4 = nn.Sequential()
        
        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2,4):
            self.to_relu_1_2.add_module(str(x), features[x])    
            
        for x in range(4,7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7,9):
            self.to_relu_2_2.add_module(str(x), features[x])  
            
        for x in range(9,12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12,14):
            self.to_relu_3_2.add_module(str(x), features[x])
        for x in range(14,16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16,18):
            self.to_relu_3_4.add_module(str(x), features[x])
            
        for x in range(18,21):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(21,23):
            self.to_relu_4_2.add_module(str(x), features[x])
        for x in range(23,25):
            self.to_relu_4_3.add_module(str(x), features[x])
        for x in range(25,27):
            self.to_relu_4_4.add_module(str(x), features[x])
    
            
        for x in range(27,30):
            self.to_relu_5_1.add_module(str(x), features[x])
        for x in range(30,32):
            self.to_relu_5_2.add_module(str(x), features[x]) 
        for x in range(32,34):
            self.to_relu_5_3.add_module(str(x), features[x])
        for x in range(34,36):
            self.to_relu_5_4.add_module(str(x), features[x])             
            
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        f = self.to_relu_1_1(x)
        f_relu_1_1 = f
        f = self.to_relu_1_2(f)
        f_relu_1_2 = f
        
        f = self.to_relu_2_1(f)
        f_relu_2_1 = f
        f = self.to_relu_2_2(f)
        f_relu_2_2 = f
        
        f = self.to_relu_3_1(f)
        f_relu_3_1 = f
        f = self.to_relu_3_2(f)
        f_relu_3_2 = f
        f = self.to_relu_3_3(f)
        f_relu_3_3 = f
        f = self.to_relu_3_4(f)
        f_relu_3_4 = f
        
        
        f = self.to_relu_4_1(f)
        f_relu_4_1 = f
        f = self.to_relu_4_2(f)
        f_relu_4_2 = f
        f = self.to_relu_4_3(f)
        f_relu_4_3 = f
        f = self.to_relu_4_4(f)
        f_relu_4_4 = f
        
        
        f = self.to_relu_5_1(f)
        f_relu_5_1 = f
        f = self.to_relu_5_2(f)
        f_relu_5_2 = f        
        f = self.to_relu_5_3(f)
        f_relu_5_3 = f
        f = self.to_relu_5_4(f)
        f_relu_5_4 = f
        
        return [f_relu_1_2, f_relu_2_2, f_relu_3_3,f_relu_4_3]