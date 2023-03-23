import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from GeneralDataset import DataSetRead
import trainer   
import tester
import utils





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
No_Classes = 6 
batchsize = 8




P_train_image = "SourceIntargetTrainImagePath"
P_val_image   = "SourceIntargetValImagePath"
V_train_image = "TargetIntargettrainImagePath"
V_val_image   = "TargetIntargetValImagePath"

    
P_test_image  = "SourceIntargetTestImagePath" 
V_test_image  = "TargetIntargetTestImagePath" 

P_train_gt    = "sourceTrainGTpath"
P_val_gt      = "sourceValGTpath" 
P_test_gt     = "sourceTestGTpath" 
V_train_gt    = "TargetTrainGTpath" 
V_val_gt      = "TargetValGTpath" 
V_test_gt     = "TargetTestGTpath" 



model_path = "Model.pth"  
n_epochs = 100 
patience = 10

model = smp.Unet(encoder_name='resnet101', 
                 encoder_depth=5,
                 encoder_weights='imagenet', 
                 decoder_use_batchnorm=True,
                 decoder_channels=(256, 128, 64, 32, 16),
                 decoder_attention_type=None, 
                 in_channels=3, 
                 classes=No_Classes, 
                 activation=None, 
                 aux_params=None)


source_trainDS = DataSetRead(V_train_image,V_train_gt)
trainDL = DataLoader(source_trainDS, batch_size=batchsize, shuffle=True, sampler=None,
                                      batch_sampler=None, num_workers=0, collate_fn=None,
                                      pin_memory=False, drop_last=False , timeout=0,
                                      worker_init_fn=None)

source_valDS = DataSetRead(V_val_image,V_val_gt)
valDL = DataLoader(source_valDS, batch_size=batchsize, shuffle=True, sampler=None,
                                      batch_sampler=None, num_workers=0, collate_fn=None,
                                      pin_memory=False, drop_last=False , timeout=0,
                                      worker_init_fn=None)

target_testDS = DataSetRead(P_test_image,P_test_gt)
testDL = DataLoader(target_testDS, batch_size=batchsize, shuffle=True, sampler=None,
                                      batch_sampler=None, num_workers=0, collate_fn=None,
                                      pin_memory=False, drop_last=False , timeout=0,
                                      worker_init_fn=None)



criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr= 1e-3, betas=(0.5, 0.999), weight_decay=5e-4)
Lr_scheduler= torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=n_epochs,steps_per_epoch=len(trainDL))



# train the SemSegm model
train_loss, valid_loss =trainer.train_model(device, n_epochs, model, trainDL, valDL, patience, criterion, optimizer, Lr_scheduler, model_path)



#Plot the loss values  
utils.plotLoss(train_loss,valid_loss , model_path)


# test part 
#Load the traind model 
model.load_state_dict(torch.load("./" + model_path))

#Evaluation the Testset Quantitative  result
tester.testmodel(device, model, testDL, No_Classes)

#Evaluation the Testset Visualization results
utils.EvalutionResult(5,target_testDS, device, model)




