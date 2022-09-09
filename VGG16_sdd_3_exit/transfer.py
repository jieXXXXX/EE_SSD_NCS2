import torch
import torch.onnx
import os
from model import SSD300, MultiBoxLoss
from model_3_sub import SSD300_backbone0,SSD300_backbone1,SSD300_backbone2,SSD300_sub1,SSD300_sub2
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
 
    model.load_state_dict(checkpoint['model_state_dict']) #初始化权重
    model = model.to(device)
    model.eval()
    
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names,
                      #dynamic_axes=dynamic_axes,
                      opset_version=10) #指定模型的输入，以及onnx的输出路径
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
                      opset_version=10) #指定模型的输入，以及onnx的输出路径
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
                      opset_version=10) #指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")
 


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    
    input = torch.randn(batch_size, 3, 300, 300)
   
    
    model = SSD300_backbone0(n_classes=n_classes) #import model
    checkpoint = torch.load('./outputs/seperate_model/backbone0.pth')
    onnx_path = './outputs/seperate_model/backbone0.onnx'
    pth_to_onnx_1_to_1(model,(input), checkpoint, onnx_path)
    
    input = torch.randn(batch_size, 256, 38, 38)
    model = SSD300_backbone1(n_classes=n_classes) #import model
    checkpoint = torch.load('./outputs/seperate_model/backbone1.pth')
    onnx_path = './outputs/seperate_model/backbone1.onnx'
    pth_to_onnx_1_to_2(model,(input), checkpoint, onnx_path)
    
    model = SSD300_sub1(n_classes=n_classes) #import model
    checkpoint = torch.load('./outputs/seperate_model/sub1.pth')
    onnx_path = './outputs/seperate_model/sub1.onnx'
    pth_to_onnx_1_to_2(model,(input), checkpoint, onnx_path)
    
    input1 = torch.randn(batch_size, 512, 38, 38)
    input2 = torch.randn(batch_size, 512, 19, 19)
    model = SSD300_backbone2(n_classes=n_classes)#import model
    checkpoint = torch.load('./outputs/seperate_model/backbone2.pth')
    onnx_path = './outputs/seperate_model/backbone2.onnx'
    pth_to_onnx_2_to_2(model,(input1,input2), checkpoint, onnx_path)
    
    model = SSD300_sub2(n_classes=n_classes) #import model
    checkpoint = torch.load('./outputs/seperate_model/sub2.pth')
    onnx_path = './outputs/seperate_model/sub2.onnx'
    pth_to_onnx_2_to_2(model,(input1,input2), checkpoint, onnx_path)
    
     