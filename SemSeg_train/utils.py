import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# helper function for data visualization
def imageshow(**images):

    """Plot images in one row with name of first string befor = 
       image is 0 to 255 with HxWxCh size . 
       gt is indces 0 to 5 with 512x512 and display as a color image""" 
    n = len(images)
    plt.figure(figsize=(10, 30))
    for i, (name, image) in enumerate(images.items()):
        # image = T.ToPILImage()(image)
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap='gray', vmin=0, vmax=4)#take 512x512 result grayscale image 
        #plt.imshow(image)   # image takes value from 0 to 1 
    plt.show()

# visualize the loss as the network trained
def plotLoss(train_loss,valid_loss, model_path):
   
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, max(valid_loss)) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(model_path.replace('.pth', '') + '_Loss.png', bbox_inches='tight')

def EvalutionResult (number_image, testDS, device, model ):

    model.to(device)
# take some test smaple
    for i in range(number_image):
        n = np.random.choice(len(testDS)-1)
        
        image, gt_mask = testDS[n] 
    
        image_vis =image
        image_vis = image.permute(1,2,0)#make image 512x512x3 
        image_vis = image_vis.numpy()*255 #un normlaize 
        image_vis = image_vis.astype('uint8')
        
        image = image.to(device).unsqueeze(0)#change the image shape to [1 ,3,512,512] to device
        pr_mask = model.predict(image)#predice the model with this image
        m = nn.Softmax(dim=1)#to convert the out to probability 
        pr_probs = m(pr_mask)              
        pr_mask = torch.argmax(pr_probs, dim=1).squeeze(1)#get max index(class) 1x512x512
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())#get numpay from tensor with size 512x512    
        imageshow(
            image= image_vis,
            ground_truth=gt_mask, 
            predicted   =pr_mask)

