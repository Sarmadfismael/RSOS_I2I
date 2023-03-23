import numpy as np
import torch
from pytorchtools import EarlyStopping
import time 
from tqdm import tqdm

def train_model(device, n_epochs, model,train_loader, val_loader, patience, criterion, optimizer, scheduler, model_path):   
    torch.cuda.empty_cache()
    
    
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True,  path=model_path)
    
    
    
    model.to(device)
    
    fit_time = time.time()
    
    for epoch in range(1, n_epochs + 1):
        
       # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        
        since = time.time()
        
        train_loss = 0
        
        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for i,(image, ground_truth)  in enumerate(tqdm(train_loader, desc='Epoch: {}/{}'.format(epoch, n_epochs))):
            
            #training phase
            image = image.to(device)
            ground_truth = ground_truth.to(device)#should have [batch, H, W] 
            #reset gradient
            optimizer.zero_grad() 
            #forward
            Pred = model(image)  #should have [batch, num_classes, H, W]
            loss = criterion(Pred, ground_truth)
            
            #backward
            loss.backward()
            #update weight 
            optimizer.step()         
                       
            #step the learning rate
            scheduler.step()
            
            # record training loss       
            train_losses.append(loss.item())
            
        
        ######################    
        # validate the model #
        ###################### 
        model.eval() # prep model for evaluation
        with torch.no_grad():
             for i, (image, ground_truth) in enumerate(tqdm(val_loader, desc='Epoch: {}/{}'.format(epoch, n_epochs))):
            
                image = image.to(device)
                ground_truth = ground_truth.to(device)
                Pred = model(image)
                #loss
                loss = criterion(Pred, ground_truth)
                valid_losses.append(loss.item())
        
        
        
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        

 
        
        print(    "Train_loss: {:.3f}..".format(train_loss),
                  "Val_Loss: {:.3f}..".format(valid_loss),
                  "Time: {:.2f}m".format((time.time()-since)/60))

            
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    
    # load the last checkpoint with the best model
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    

    return avg_train_losses, avg_valid_losses