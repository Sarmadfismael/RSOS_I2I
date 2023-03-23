This repository contains the official implementation of the 
# "Unsupervised Domain Adaptation for the Semantic Segmentation of Remote Sensing Images via One-Shot Image-to-Image Translation"
in https://arxiv.org/abs/2212.03826 (to be submitted in GRSL 2023). 

# Requirements: 
### - Pytorch 1.11.0 
###  * python 3.8.10
###  + Semantic segmentation model (Smp V0.2.1)


# Dataset:
SPRS dataset: Potsdam IRRG and Vaihingen IRRG with size (512\times512)


# train:

To train the I2I translation model (step 1), use the I2I_train folder. 
Then, to train the Semantic segmentation model (step 2), use the  SemSeg_train folder.
The P_IRRG target one-shot images and V_IRRG target one shot images contain the one-shot image that has been used during the train g of the I2IT


# test and evaluate: 
Use the last three line  of the SemSeg_train.py 
