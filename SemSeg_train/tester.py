from Metrics import SemSeg_Metric
import torch 
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

def testmodel(device, model, testDL, No_Classes) :

    evaluator = SemSeg_Metric(No_Classes)
    model.to(device)
    model.eval()
    evaluator.reset()


    for i,(image, target)  in enumerate(tqdm(testDL)):
    
        
        image =image.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(image)
            
            
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)
        

        

    mIoU = evaluator.Intersection_over_Union()
    f1_score = 2 * (evaluator.precision() * evaluator.recall())  / ((evaluator.precision() + evaluator.recall())+ 1e-7)

    O_PA = evaluator.Pixel_Accuracy()
    O_mIoU = np.nanmean(mIoU)
    O_f1_score = np.nanmean(f1_score)
    
    result_table = [['','Clutter/background', 'Building', 'Low vegetation', 'Tree', 'Car','Impervious surfaces','Average'], 
               ['f1_score',np.round(f1_score[0]*100,2),np.round(f1_score[1]*100,2),np.round(f1_score[2]*100,2),np.round(f1_score[3]*100,2),np.round(f1_score[4]*100,2),np.round(f1_score[5]*100,2),np.round(O_f1_score*100,2)],
               ['mIoU',np.round(mIoU[0]*100,2),np.round(mIoU[1]*100,2),np.round(mIoU[2]*100,2),np.round(mIoU[3]*100,2),np.round(mIoU[4]*100,2),np.round(mIoU[5]*100,2),np.round(O_mIoU*100,2)],
               ['Overall_pixel_accurecy','','','','','','', np.round(O_PA*100,2)]]

    print(tabulate(result_table,  tablefmt='grid'))
