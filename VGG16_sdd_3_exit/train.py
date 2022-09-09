import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import warnings
from datetime import datetime as dt4

from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *

warnings.filterwarnings("ignore", category=UserWarning)#ignore some unnessary information

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?
save_path = 'outputs/'


# Model parameters
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
workers = 1  # number of workers for loading data in the DataLoader
print_freq = 30  # print training status every __ batches
lr = 1e-3 # learning rate
decay_lr_at = [500,600]  # decay learning rate after these many iterations
decay_lr_to = 0.5  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum 
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
backbone_epochs=10
joint_epochs = 10
cudnn.benchmark = True
 
def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at
    

    # Move to default device
    model = SSD300(n_classes=n_classes)
    
    #copy the weight from pretrained model 
    """
    model_weight_path = "./checkpoint_ssdbest.pth"
    pre_weights = torch.load(model_weight_path, map_location='cuda') 
    pre_dict = {k: v for k, v in pre_weights.items()}
    missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)
    """
    
    #set the device and the loss function
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)#loss criterion
    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
   
    

    #training with seperate method
    Train_joint(model, train_loader,save_path, criterion,optimizer=None,
                backbone_epochs=backbone_epochs, joint_epochs=joint_epochs,
                pretrain_backbone=True)

    



def Train_backbone(model, train_loader, save_path,criterion,epochs=50,
                   optimizer=None,):
    #train network backbone
    #-----------------------set the optimizer of backbone-----------------------
    if optimizer is None:
        #set to backbone optimizer, only update the backbone weight     
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias') and ('backbone' in param_name) :
                    biases.append(param)
                elif param_name.endswith('.bias') and ('exitF_aux_convs' in param_name):
                    biases.append(param)
                elif param_name.endswith('.bias') and ('exitF_pred_convs' in param_name):
                    biases.append(param)
                else:
                    not_biases.append(param)
        
        optimizer = optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], 
                              lr=lr,momentum=momentum, weight_decay=weight_decay)
        
        
    #-------------------------------start traning backbone---------------------
    
    for epoch in range(epochs):
        model.train()
        #initial print information
        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        loss_average = AverageMeter()  # loss
        start = time.time()
       
        #---------------Decay learning rate at particular epochs---------------
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)
        
        #-------------------training loop-------------------
        for i, (images, boxes, labels, _) in enumerate(train_loader):
            data_time.update(time.time() - start)

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
                
            # Forward prop.
            predicted_locs, predicted_scores = model(images,train_bb=True)  # (N, 8732, 4), (N, 8732, n_classes)
          
            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            
            # Backward prop.
            optimizer.zero_grad()
            loss.backward()
        
            # Clip gradients, if necessaryu
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
        
            # Update model
            optimizer.step()
         
            #print shwo result ,calculate time
            loss_average.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)
            start = time.time()
           
            # Print status
            if i % print_freq == 0:
                print(
                      'Epoch: [{0}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch,
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, 
                                                                      loss=loss_average))
        # free some memory since their histories may be stored        
        del predicted_locs, predicted_scores, images, boxes, labels  
        #-----------------------save model file------------------------------
        savepoint = Save_model(model, save_path, file_prefix='backbone', opt=optimizer)
        model_dict=model.state_dict()        
        
    return savepoint
      
   
def Train_joint(model, train_loader, save_path, criterion,optimizer=None,
                backbone_epochs=50,
                joint_epochs=100, pretrain_backbone=True):
    
    #inital print data information
    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    loss_average = AverageMeter()  # loss
    start = time.time()
    
    #----------------------set pretrain backbone if pretrain_backbone==True -------------------------
    if pretrain_backbone:
        print("PRETRAINING BACKBONE FROM SCRATCH")
        folder_path = 'pre_Trn_bb_' + timestamp
        #folder_path = 'pre_Trn_bb_'
        best_bb_path = Train_backbone(model, train_loader, os.path.join(save_path, folder_path),
                                      criterion,epochs=backbone_epochs,optimizer=None)
        
        #train the rest...
        print("LOADING BEST BACKBONE")
        Load_model(model, best_bb_path)
     
        
        print("JOINT TRAINING WITH PRETRAINED BACKBONE")

        prefix = 'pretrn-joint'
        
    else:
        #jointly trains backbone and exits from scratch
        print("JOINT TRAINING FROM SCRATCH")
        #folder_path = 'jnt_fr_scrcth' + timestamp
        folder_path = '3_exit_with_pretraining_model'
        prefix = 'joint'

    spth = os.path.join(save_path, folder_path)
    
    #-----------------------------freeze the backbone model---------------------------
    
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if  ('backbone' in param_name) :
               param.requires_grad=False
            elif ('exitF_aux_convs' in param_name):
                param.requires_grad=False
            elif ('exitF_pred_convs' in param_name):
                param.requires_grad=False
            else:
                param.requires_grad=True
    
    #------------------------------train joint optimizer set------------------
    #set up the joint optimiser
    if optimizer is None: 
        biases = list()
        not_biases = list()
        
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias') and ('exits.0'in param_name):
                    biases.append(param)
                else:
                    not_biases.append(param)
                   
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    #---------------------------start trainning exit with backbone--------------------
    for epoch in range(joint_epochs):
        model.train()
        #-------------------training loop-------------------
        for i, (images, boxes, labels, _) in enumerate(train_loader):
            
            data_time.update(time.time() - start)

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
                
            # Forward prop.
            predicted_locs, predicted_scores = model(images,train_bb=False)  # (N, 8732, 4), (N, 8732, n_classes)
             
            # Loss
            losses = [weighting * (criterion(predicted_loc, predicted_score, boxes, labels))
                        for weighting, predicted_loc,predicted_score in zip(model.exit_loss_weights,predicted_locs,predicted_scores)]
        
            optimizer.zero_grad()
            
            for loss in losses[:-1]:
                loss.backward()
                loss_average.update(loss.item(), images.size(0))    
          
            # Clip gradients, if necessary          
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
           
            # Update model
            optimizer.step()
            
            #print shwo result ,calculate time
            batch_time.update(time.time() - start)
            start = time.time()
            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, 
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, 
                                                                      loss=loss_average))
        
            
        #-----------------------save model file------------------------------ 
        savepoint = Save_model(model, spth, "joint_weight",opt=optimizer)
        model_dict = model.state_dict()
        del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
   
    return  

 

if __name__ == '__main__':
    main()
