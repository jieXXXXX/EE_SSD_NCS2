import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
from torch import nn
from model import SSD_mobilenetv2,MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
from datetime import datetime as dt
import torchvision.models.mobilenet
import warnings
import numpy

import mobilenetv1_ssd_config

warnings.filterwarnings("ignore", category=UserWarning)#ignore some unimportant warning messages

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
workers = 1  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3 # learning rate
decay_lr_at = [1700,1800]  # decay learning rate after these many iterations
decay_lr_to = 0.5  # decay learning rate to this fracti on of the existing learning rate
momentum = 0.9  # momentum 
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
backbone_epochs=20
joint_epochs = 100
  
save_path = 'outputs/'
cudnn.benchmark = True
 
def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at
    
   
    # Move to default device
    #model = create_mobilenetv2_ssd_lite(n_classes )
    model = SSD_mobilenetv2(num_classes = n_classes)
    config = mobilenetv1_ssd_config

    # ------------------------load pretrain weights--------------------------
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth

    model_weight_path = "./mb2-ssd-lite-mp-0_686.pth"
    pre_weights = torch.load(model_weight_path, map_location='cuda')
        
    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items()}
   
    for param_name in pre_dict.keys():
        print(param_name)
        param =pre_dict[param_name]
        if('num_batches_tracked' in param_name):
            continue
            
        if('base_net' in param_name):  
            
            if ('.17' in param_name) :
                param_name_s = param_name.split('.', 2 )
                model.backbone[2].features[0].state_dict()[param_name_s[2]].copy_(param)
            elif  ('.18' in param_name) :
                param_name_s = param_name.split('.', 2 )
                model.backbone[2].features[1].state_dict()[param_name_s[2]].copy_(param)
            
            elif  ('.14' in param_name) :
                param_name_s = param_name.split('.', 2 )
                model.backbone[1].features[0].state_dict()[param_name_s[2]].copy_(param)
            elif  ('.15' in param_name) :
                param_name_s = param_name.split('.', 2 )
                model.backbone[1].features[1].state_dict()[param_name_s[2]].copy_(param)
            elif  ('.16' in param_name) :
               param_name_s = param_name.split('.', 2 )
               model.backbone[1].features[2].state_dict()[param_name_s[2]].copy_(param)
            
            else:
                param_name_s = param_name.split('.', 1 )
                model.backbone[0].features.state_dict()[param_name_s[1]].copy_(param)
                  
        else:
            model.exits[-1].state_dict()[param_name].copy_(param)
           
    #----------------------model device and loss function seting------------------
    model = model.to(device)
    #summary(model, (3, 300,300))
    criterion = MultiBoxLoss(priors_cxcy=config.priors.to(device)).to(device)
    #--------------------------load datasets---------------------------------
    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
   
    #----------------------training---------------------------------------------
    Train_joint(model, train_loader,save_path, criterion,optimizer=None,
                backbone_epochs=backbone_epochs, joint_epochs=joint_epochs,
                pretrain_backbone=True)




def Train_backbone(model, train_loader, save_path,criterion,epochs=50,
                   optimizer=None,):
    #train network backbone
    
    #-----------------------set the optimizer of backbone-----------------------
    if optimizer is None:
        
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias') and ('backbone' in param_name) :
                    biases.append(param)
                elif param_name.endswith('.bias') and ('exits.2' in param_name):
                    biases.append(param)
                else:
                    not_biases.append(param)
                
      
        optimizer = optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], 
                              lr=lr,momentum=momentum, weight_decay=weight_decay)
        
        
    #-------------------------------start traning backbone---------------------------
    
    for epoch in range(epochs):
        
        model.train()
        
        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        loss_average = AverageMeter()  # loss

        start = time.time()

        #---------------Decay learning rate at particular epochs-------------------
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)
            
        #print(epoch+1, flush=True)
        
        #-------------------training loop-------------------
        
        for i, (images, boxes, labels, _) in enumerate(train_loader):
            acc=0.0
            data_time.update(time.time() - start)
            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
         
            # Forward prop.
            predicted_scores,predicted_locs = model(images) 
           
            # Loss            
            loss = criterion(predicted_locs[-1], predicted_scores[-1], boxes, labels)
            
            # Backward prop.
            optimizer.zero_grad()
            loss.backward()
        
            # Clip gradients, if necessaryu
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
        
            # Update model
            optimizer.step()
            
           
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
                                                                      loss=loss_average
                                                                     ))
             
        del  predicted_scores, images,  labels  # free some memory since their histories may be stored

        #-----------------------save model file------------------------------      
        savepoint = Save_model(model, save_path, file_prefix='backbone'+str(epoch), opt=optimizer)
        model_dict=model.state_dict()
        
    return savepoint
      
   
def Train_joint(model, train_loader, save_path, criterion,optimizer=None,
                backbone_epochs=50,
                joint_epochs=100, pretrain_backbone=True):
    
    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    loss_average = AverageMeter()  # loss

    start = time.time()
    
    #----------------------set pretrain backbone -------------------------
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
        folder_path = 'training with pretrained weight file' + timestamp
        prefix = 'joint'

    spth = os.path.join(save_path, folder_path)
    
    #-----------------------------freeze the backbone model---------------------------
    
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if  ('backbone' in param_name) :
               param.requires_grad=False
            elif ('exits.2' in param_name):
                param.requires_grad=False
            else:
                param.requires_grad=True
    
    #------------------------------train joint optimizer set------------------
    #set up the joint optimiser
    if optimizer is None: #TODO separate optim function to reduce code, maybe pass params?
        #set to branchynet default
        biases = list()
        not_biases = list()
        
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias') and ('exits.0'in param_name):
                    biases.append(param)
                elif param_name.endswith('.bias') and ('exits.1'in param_name):
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
            predicted_scores,predicted_locs= model(images)  # (N, 3000, 4), (N, 3000, n_classes)
            
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
        save_checkpoint(epoch, model, optimizer)
        savepoint = Save_model(model, spth, "joint_weight"+str(epoch),opt=optimizer)
        model_dict = model.state_dict()
     
        
        del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
   
    return  

 

if __name__ == '__main__':
    main()
