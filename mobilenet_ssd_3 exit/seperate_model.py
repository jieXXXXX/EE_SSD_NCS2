from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
from model import SSD_mobilenetv2,MultiBoxLoss
import mobilenetv1_ssd_config
from model_3_sub.py import SSD_mobilenetv2_backbone1,SSD_mobilenetv2_backbone2,SSD_mobilenetv2_backbone3,SSD_mobilenetv2_exit1,SSD_mobilenetv2_exit2
import warnings

warnings.filterwarnings("ignore", category=UserWarning)#ignore some unimportant warning messages

md_pth="joint_weight.pth"
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
workers = 1  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
momentum = 0.9  # momentum 
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
save_path = 'outputs/seperate_model/'


model = SSD_mobilenetv2(num_classes=n_classes)
checkpoint = torch.load(md_pth)

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

model_backbone1  = SSD_mobilenetv2_backbone1(num_classes=n_classes)
model_backbone1 = model_backbone1.to(device) 

model_backbone2  = SSD_mobilenetv2_backbone2(num_classes=n_classes)
model_backbone2 = model_backbone2.to(device) 

model_backbone3  = SSD_mobilenetv2_backbone3(num_classes=n_classes)
model_backbone3 = model_backbone3.to(device) 

model_sub1 = SSD_mobilenetv2_exit1(num_classes=n_classes)
model_sub1 = model_sub1.to(device)

model_sub2 = SSD_mobilenetv2_exit2(num_classes=n_classes)
model_sub2 = model_sub2.to(device)

def Save_model(model, path, file_prefix='', seed=None, epoch=None, opt=None, loss=None):
    #TODO add saving for inference only
    #TODO add bool to save as onnx - remember about fixed bs
    #saves the model in pytorch format to the path specified
    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")
    filenm = file_prefix  
    save_dict ={'timestamp': timestamp,
                'model_state_dict': model.state_dict()
                }

    if seed is not None:
        save_dict['seed'] = seed
    if epoch is not None:
        save_dict['epoch'] = epoch
        filenm += f'{epoch:03d}'
    if opt is not None:
        save_dict['opt_state_dict'] = opt.state_dict()
        #save_dict['opt_state_dict'] = opt.state_dict()
    if loss is not None:
        save_dict['loss'] = loss

    if not os.path.exists(path):
        os.makedirs(path)

    filenm += '.pth'
    file_path = os.path.join(path, filenm)
    torch.save(save_dict, file_path)
    
    #print("Saved to:", file_path)
    return file_path

#---------------------------optimizer setting----------------------------------
biases = list()
not_biases = list()

for param_name, param in model.named_parameters():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
        else:
            not_biases.append(param)
            
optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                            lr=lr, momentum=momentum, weight_decay=weight_decay)
 
#--------------------------------seprate the model----------------------------     
for param_name in model.state_dict().keys():

        param = model.state_dict()[param_name]
        
        if ('backbone.0' in param_name) :
            model_backbone1.state_dict()[param_name].copy_(param)
        elif ('backbone.1' in param_name):
            model_backbone2.state_dict()[param_name].copy_(param)
        elif ('backbone.2' in param_name):
           model_backbone3.state_dict()[param_name].copy_(param)
        elif ('exits.0' in param_name):
           model_sub1.state_dict()[param_name].copy_(param)
        elif('exits.1' in param_name):
           model_sub2.state_dict()[param_name].copy_(param)
        elif('exits.2' in param_name):
           model_backbone3.state_dict()[param_name].copy_(param)
                
        else:
             print("-----------")
             print(param_name)
        
#----------------------------------save the model--------------------------  
Save_model(model_backbone1, save_path, file_prefix='backbone1', opt=optimizer)        
Save_model(model_backbone2, save_path, file_prefix='backbone2', opt=optimizer)
Save_model(model_backbone3, save_path, file_prefix='backbone3', opt=optimizer)
Save_model(model_sub1, save_path, file_prefix='sub1', opt=optimizer)
Save_model(model_sub2, save_path, file_prefix='sub2', opt=optimizer)

