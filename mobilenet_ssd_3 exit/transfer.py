import torch
import torch.onnx
import onnx
import os
from model import SSD_mobilenetv2,MultiBoxLoss
import mobilenetv1_ssd_config
from model_3_sub.py import SSD_mobilenetv2_backbone1,SSD_mobilenetv2_backbone2,SSD_mobilenetv2_backbone3,SSD_mobilenetv2_exit1,SSD_mobilenetv2_exit2
from datasets import PascalVOCDataset
from utils import *
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8 

def pth_to_onnx_1_to_1(model,input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
 

    model.load_state_dict(checkpoint['model_state_dict'])#initialize weight
    model = model.to(device)
    model.eval()
  
    
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names,
                      #dynamic_axes=dynamic_axes,
                      opset_version=10,)  
    print("Exporting .pth model to onnx model has been successful!")
    
def pth_to_onnx_1_to_2(model,input, checkpoint, onnx_path, input_names=['input'], output_names=['output1','output2'], device='cpu'):    

    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
 
    model.load_state_dict(checkpoint['model_state_dict']) #初始化权重
    model = model.to(device)
    model.eval()
  
    
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names,
                      #dynamic_axes=dynamic_axes,
                      opset_version=10) 
    print("Exporting .pth model to onnx model has been successful!")
        
def pth_to_onnx_2_to_2(model,input, checkpoint, onnx_path, input_names=['input1','input2'], output_names=['output1','output2'], device='cpu'):    

    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
 
    model.load_state_dict(checkpoint['model_state_dict']) #初始化权重
    model = model.to(device)
    model.eval()
  
    
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names,
                      #dynamic_axes=dynamic_axes,
                      opset_version=10)  
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    
    input = torch.randn(batch_size, 3, 300, 300)
   
    
    model = SSD_mobilenetv2_backbone1(num_classes=n_classes) #import model
    checkpoint = torch.load('./outputs/seperate_model/backbone1.pth')
    onnx_path = './outputs/seperate_model/backbone1.onnx'
    pth_to_onnx_1_to_1(model,(input), checkpoint, onnx_path)
    
    input = torch.randn(batch_size, 96, 19, 19)
    model = SSD_mobilenetv2_backbone2(num_classes=n_classes) #import model
    checkpoint = torch.load('./outputs/seperate_model/backbone2.pth')
    onnx_path = './outputs/seperate_model/backbone2.onnx'
    pth_to_onnx_1_to_2(model,(input), checkpoint, onnx_path)
    
    model = SSD_mobilenetv2_exit1(num_classes=n_classes) #import model
    checkpoint = torch.load('./outputs/seperate_model/sub1.pth')
    onnx_path = './outputs/seperate_model/sub1.onnx'
    pth_to_onnx_1_to_2(model,(input), checkpoint, onnx_path)
    
    
    input1 = torch.randn(batch_size, 160, 10, 10)
    input2 = torch.randn(batch_size, 576, 19, 19)
    model = SSD_mobilenetv2_backbone3(num_classes=n_classes) #import model
    checkpoint = torch.load('./outputs/seperate_model/backbone3.pth')
    onnx_path = './outputs/seperate_model/backbone3.onnx'
    pth_to_onnx_2_to_2(model,(input1,input2), checkpoint, onnx_path)
    
    model = SSD_mobilenetv2_exit2(num_classes=n_classes) #import model
    checkpoint = torch.load('./outputs/seperate_model/sub2.pth')
    onnx_path = './outputs/seperate_model/sub2.onnx'
    pth_to_onnx_2_to_2(model,(input1,input2), checkpoint, onnx_path)
    