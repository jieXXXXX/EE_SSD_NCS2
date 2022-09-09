from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
from model import SSD300, MultiBoxLoss
from model_3_sub import SSD300_backbone0,SSD300_backbone1,SSD300_backbone2,SSD300_sub1,SSD300_sub2
import warnings

warnings.filterwarnings("ignore", category=UserWarning)#ignore some unnessary information

md_pth="joint_weight.pth"#spearate weight file
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device setting

lr = 1e-3  # learning rate
momentum = 0.9  # momentum 
weight_decay = 5e-4  # weight decay

save_path = 'outputs/seperate_model/'
 
#-----------------------------seprate model setting-----------------------------
model = SSD300(n_classes=n_classes)
checkpoint = torch.load(md_pth)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

model_backbone0  = SSD300_backbone0(n_classes=n_classes)
model_backbone0 = model_backbone0.to(device)

model_backbone1  = SSD300_backbone1(n_classes=n_classes)
model_backbone1 = model_backbone1.to(device)

model_backbone2  = SSD300_backbone2(n_classes=n_classes)
model_backbone2 = model_backbone2.to(device)
 
model_sub1 = SSD300_sub1(n_classes=n_classes)
model_sub1 = model_sub1.to(device)

model_sub2 = SSD300_sub2(n_classes=n_classes)
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
    if loss is not None:
        save_dict['loss'] = loss

    if not os.path.exists(path):
        os.makedirs(path)

    filenm += '.pth'
    file_path = os.path.join(path, filenm)
    torch.save(save_dict, file_path)
    
    #print("Saved to:", file_path)
    return file_path


#----------------------------------set the optimizer--------------------------
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

#---------------------------copy the seperated weight ------------------------
for param_name in model.state_dict().keys():

        param = model.state_dict()[param_name]
        
        if ('backbone.0' in param_name) :
            model_backbone0.state_dict()[param_name].copy_(param)
        elif ('backbone.1' in param_name):
            model_backbone1.state_dict()[param_name].copy_(param)
        elif ('backbone.2' in param_name):
           model_backbone2.state_dict()[param_name].copy_(param)
        elif ('exits.0' in param_name):
           model_sub1.state_dict()[param_name].copy_(param)
        elif('exits.1' in param_name):
           model_sub2.state_dict()[param_name].copy_(param)
        elif('exits.2' in param_name):
          model_backbone2.state_dict()[param_name].copy_(param)
                
        else:
             print(param_name)
#------------------------------------save model------------------------------       
Save_model(model_backbone0, save_path, file_prefix='backbone0', opt=optimizer)        
Save_model(model_backbone1, save_path, file_prefix='backbone1', opt=optimizer)
Save_model(model_backbone2, save_path, file_prefix='backbone2', opt=optimizer)
Save_model(model_sub1, save_path, file_prefix='sub1', opt=optimizer)
Save_model(model_sub2, save_path, file_prefix='sub2', opt=optimizer)

  