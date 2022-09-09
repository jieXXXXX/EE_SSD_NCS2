from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
from torchvision import transforms
import numpy as np
import math
from typing import List, Tuple
from scipy.stats import entropy
import box_utils
#from vision.ssd.ssd import SSD, GraphPath

#from vision.ssd.config import mobilenetv1_ssd_config as config
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #

n_define_class = 21 #VOC2007


# Modified from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.
# In this version, Relu6 is replaced with Relu to make it ONNX compatible.
# BatchNorm Layer is optional to make it easy do batch norm confusion.


def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            ReLU(inplace=True)
        )

def conv_1x1_bn(inp, oup, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            ReLU(inplace=True)
        )

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Mobilenetv2_Backbone1(nn.Module):
    def __init__(self, n_class=n_define_class, input_size=224, width_mult=1., dropout_ratio=0.2,
                 use_batch_norm=True, onnx_compatible=False):
        super(Mobilenetv2_Backbone1, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 96
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            #[6, 160, 3, 2],
            #[6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        
        self.features = [conv_bn(3, input_channel, 2, onnx_compatible=onnx_compatible)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                else:
                    self.features.append(block(input_channel, output_channel, 1,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                input_channel = output_channel
    
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            
class Mobilenetv2_Backbone2(nn.Module):
    def __init__(self, n_class=n_define_class, input_size=10, width_mult=1., dropout_ratio=0.2,
                 use_batch_norm=True, onnx_compatible=False):
        super(Mobilenetv2_Backbone2, self).__init__()
        block = InvertedResidual
        input_channel = 96
        last_channel = 160
        interverted_residual_setting = [
            # t, c, n, s
            #[1, 16, 1, 1],
            #[6, 24, 2, 2],
            #[6, 32, 3, 2],
            #[6, 64, 4, 2],
            #[6, 96, 3, 1],
            [6, 160, 3, 2],
            #[6, 320, 1, 1],
        ]

        # building first layer
        #assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        
        self.features = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                else:
                    self.features.append(block(input_channel, output_channel, 1,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
      
class Mobilenetv2_Backbone3(nn.Module):
    def __init__(self, n_class=n_define_class, input_size=10, width_mult=1., dropout_ratio=0.2,
                 use_batch_norm=True, onnx_compatible=False):
        super(Mobilenetv2_Backbone3, self).__init__()
        block = InvertedResidual
        input_channel = 160
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            #[1, 16, 1, 1],
            #[6, 24, 2, 2],
            #[6, 32, 3, 2],
            #[6, 64, 4, 2],
            #[6, 96, 3, 1],
            #[6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        #assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        
        self.features = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                else:
                    self.features.append(block(input_channel, output_channel, 1,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Mobilenetv2_exit1(nn.Module):

    def __init__(self, n_class=n_define_class, input_size=10, width_mult=1., dropout_ratio=0.2,
                 use_batch_norm=True, onnx_compatible=False):
        super(Mobilenetv2_exit1, self).__init__()
        
        self.num_classes = n_class
        input_channel = 96
        last_channel = 1280
        
        self.conv_exit_1 = nn.Sequential(nn.Conv2d(96 ,576, stride=1, padding=1 ,kernel_size = 3),
                                         nn.ReLU6(inplace=True))
        
        self.conv_exit_2 =  nn.Sequential(nn.Conv2d(576 ,576,kernel_size = 3, stride=2, padding=1,groups=576),
                                         nn.ReLU6(inplace=True))
        
        self.features = (conv_1x1_bn(576, last_channel,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
    
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(last_channel, n_class),
        )
        
        self._initialize_weights()
        
        self.extras = Extras()()
        
        self.classification_headers = Classification_headers()(width_mult = 1.0,num_classes =self.num_classes )
        
        self.regression_headers = Regression_headers()(width_mult = 1.0)
               
       

    def forward(self,x):
        
        confidences=[]
        locations = []
        header_index=0
        
        y_14 = self.conv_exit_1(x)
        x = self.conv_exit_2(y_14)
        x = self.features(x)
        
        
        confidence_14, location_14 = self.compute_header(header_index, y_14)   
        confidences.append(confidence_14)
        locations.append(location_14)
        header_index+=1
        
        y=x
        confidence, location = self.compute_header(header_index, y)   
        confidences.append(confidence)
        locations.append(location)
        header_index+=1         
        
        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
            
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        return confidences,locations
    
    def compute_header(self, i, x):
        
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        
        return confidence,location
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
class Mobilenetv2_exit2(nn.Module):

    def __init__(self, n_class=n_define_class, input_size=10, width_mult=1., dropout_ratio=0.2,
                 use_batch_norm=True, onnx_compatible=False):
        super(Mobilenetv2_exit2, self).__init__()
        
        self.num_classes = n_class
        input_channel = 160
        last_channel = 1280
        
        
        
        self.features = (conv_1x1_bn(input_channel, last_channel,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(last_channel, n_class),
        )
        
        self._initialize_weights()
        
        self.extras = Extras()()
        
        self.classification_headers = Classification_headers()(width_mult = 1.0,num_classes =self.num_classes )
        
        self.regression_headers = Regression_headers()(width_mult = 1.0)
       

    def forward(self,x,y_14):
        
        confidences=[]
        locations = []
        header_index=0
        confidence_14, location_14 = self.compute_header(header_index, y_14)   
        confidences.append(confidence_14)
        locations.append(location_14)
        header_index+=1
        
        x = self.features(x)
        y=x
        confidence, location = self.compute_header(header_index, y)   
        confidences.append(confidence)
        locations.append(location)
        header_index+=1         
        
        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
            
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        return confidences,locations
    
    def compute_header(self, i, x):
        
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        
        return confidence,location
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
 
class Mobilenetv2_exixFinal(nn.Module):

    def __init__(self, n_class=n_define_class):
        super(Mobilenetv2_exixFinal, self).__init__()
        
        self.num_classes = n_class
    
        self._initialize_weights()    
    
        self.extras = Extras()()
        
        self.classification_headers = Classification_headers()(width_mult = 1.0,num_classes =self.num_classes )
        
        self.regression_headers = Regression_headers()(width_mult = 1.0)

    def forward(self,x,y_14):
        
        confidences=[]
        locations = []
        
        confidence_14, location_14 = self.compute_header(0, y_14)
        confidence_19, location_19 = self.compute_header(1, x)
        confidences.append(confidence_14)
        confidences.append(confidence_19)
        locations.append(location_14)
        locations.append(location_19)
      
        header_index=2
        
        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
            
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        return confidences,locations
    
    def compute_header(self, i, x):
        
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        
        return confidence,location
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
 
class Extras(nn.Module):                
    def __init__(self,):
        super(Extras, self).__init__()
        
    def forward(self):
        extras = nn.ModuleList([
            InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
            InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
            InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
            InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
        ])
        return extras

class Regression_headers(nn.Module):                
    def __init__(self,):
        super(Regression_headers, self).__init__()
        
    def forward(self,width_mult):
        
        regression_headers = nn.ModuleList([
            SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * 4,
                            kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=1280, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            nn.Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
        ])

        return regression_headers

class Classification_headers(nn.Module):                
    def __init__(self,):
        super(Classification_headers, self).__init__()
        
    def forward(self,width_mult,num_classes):
        
        classification_headers = nn.ModuleList([
            SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=1280, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
        ])
        return classification_headers


class SSD_mobilenetv2(nn.Module):
    def __init__(self, num_classes: int,exit_threshold=0.5, is_test=False, config=None, device=None):
        
        super(SSD_mobilenetv2, self).__init__()

        self.num_classes = num_classes
        self.exit_threshold = torch.tensor([exit_threshold], dtype=torch.float32)
        self.exit_loss_weights = [1.0,1.0,1.0]
        self.fast_inference_mode = False
        self.source_layer_indexes =  source_layer_indexes = [GraphPath(0, 'conv', 3)
                                                             , 19,
                                                             ]
        
        #contruct the network
        self.backbone = nn.ModuleList()
        self.exits = nn.ModuleList()  
        self._build_backbone()
        self._build_exits()
        
   
        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
     
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
    
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        confidence_14=0
        location_14=0
        y_14 = torch.Tensor()
        
        if self.fast_inference_mode: #inference with early exit
            for  bb, ee in zip(self.backbone, self.exits):
                
                     
                if("Mobilenetv2_Backbone1") in str(bb):
                    x = bb(x)
                    confidence,location = ee(x)
                    confidences.append(confidence)
                    locations.append(location)
                    
                    if self.exit_criterion_entropy(confidence,location):
                       print("-----------first early eixt -----------------")
                       #return confidence, location   
                
                if("Mobilenetv2_Backbone2") in str(bb):
                    base_net = bb.features
      
                    end_layer_index = self.source_layer_indexes[0]
                    path = end_layer_index
                    end_layer_index = end_layer_index.s0
                    added_layer = None
                    
                    sub = getattr(base_net[end_layer_index], path.name)
                    for layer in sub[:path.s1]:
                        x = layer(x)
                    y = x
                    for layer in sub[path.s1:]:
                        x = layer(x)
                    end_layer_index += 1
                    start_layer_index = end_layer_index
                    y_14 = y 
                    
                    for layer in base_net[start_layer_index:]:
                        x = layer(x)
                        
                    confidence,location = ee(x,y_14)
                    confidences.append(confidence)
                    locations.append(location)
                    if self.exit_criterion_entropy(confidence,location):
                        print("-----------second early eixt-----------------")
                        return confidence, location
                    
                   
                
                if("Mobilenetv2_Backbone3") in str(bb):
                    base_net = bb.features
                    
                    for layer in base_net:
                        x = layer(x)
                    confidence,location = ee(x,y_14)
                    
                    return confidence, location
                
        
        else:#training
            for  bb, ee in zip(self.backbone, self.exits):
                 
                if("Mobilenetv2_Backbone1") in str(bb):
                    #backbone[first],exits[first]
                    x = bb(x)
                    confidence,location = ee(x)
                    confidences.append(confidence)
                    locations.append(location)
                    
                if("Mobilenetv2_Backbone2") in str(bb):
                    #backbone[second],exits[second]
                    base_net = bb.features
      
                    end_layer_index = self.source_layer_indexes[0]
                    path = end_layer_index
                    end_layer_index = end_layer_index.s0
                    added_layer = None
                    
                    sub = getattr(base_net[end_layer_index], path.name)
                    for layer in sub[:path.s1]:
                        x = layer(x)
                    y = x
                    for layer in sub[path.s1:]:
                        x = layer(x)
                    end_layer_index += 1
                    start_layer_index = end_layer_index
                    y_14 = y
                    
                    for layer in base_net[start_layer_index:]:
                        x = layer(x)
                        
                    confidence,location = ee(x,y_14)
                    confidences.append(confidence)
                    locations.append(location)
                
                if("Mobilenetv2_Backbone3") in str(bb):
                    #backbone[final],exits[final]
                    base_net = bb.features
                    for layer in base_net:
                        x = layer(x)
                    
                    confidence,location = ee(x,y_14)
                    confidences.append(confidence)
                    locations.append(location)
                    
        return confidences, locations
   
    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)
 
    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
        
    def _build_backbone(self):
       
        strt_bl = Mobilenetv2_Backbone1()#NOTE was padding 3
        
        self.backbone.append(strt_bl)
        
        bb_layers = []
        bb_layers.append(Mobilenetv2_Backbone2())
        self.backbone.append(*bb_layers)
        
        bb_layers2 = []
        bb_layers2.append(Mobilenetv2_Backbone3())
        self.backbone.append(*bb_layers2)
        
    def _build_exits(self):
        #adding early exits/branches
        #early exit 1
        ee1 = Mobilenetv2_exit1()#
        self.exits.append(ee1)
        ee2 = Mobilenetv2_exit2()
        self.exits.append(ee2)
        #final exit
        eeF = Mobilenetv2_exixFinal()
        self.exits.append(eeF)

  
    def set_fast_inf_mode(self, mode=True):
    
        if mode:
            self.eval()
        self.fast_inference_mode = mode
    
    def exit_criterion_top1(self, prediction,location): #NOT for batch size > 1
        #evaluate the exit criterion on the result provided
        #return true if it can exit, false if it can't
        with torch.no_grad():
            exit_bool=False
            prediction = F.softmax(prediction, dim=2)
            max_prediction=np.max(np.array((prediction[0][:,1:-1]).cpu()))#要剔除背景一类的预测值
            batch_size = location.size(0)
            for i in range(batch_size):
                if (max_prediction) > (self.exit_threshold).cuda() :
                    exit_bool=True
                    break
                    
        return exit_bool
    
    def exit_criterion_entropy(self, prediction,location): #NOT for batch size > 1
        #evaluate the exit criterion on the result provided
        #return true if it can exit, false if it can't
        location = torch.tensor(location) 
        prediction =  torch.tensor(prediction) 
        prediction = F.softmax(prediction, dim=2)
        n_labels = len(prediction)-1 #not include background
        np.array((prediction[0][:,1:-1]).cpu())
        for i in range(prediction.shape[1]):
            entropy_value = entropy((prediction[0][i][1:-1]).cpu())
            
            exit_bool=False
            batch_size = location.size(0)
            for i in range(batch_size):
                if ((entropy_value)) < (self.exit_threshold) :
                    exit_bool=True        
                    print("entropy",entropy_value)
                    return exit_bool
        return exit_bool
    
    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k,priors_cxcy):
        """
        Decipher the 3000 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 3000 prior boxes, a tensor of dimensions (N, 3000, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 3000, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 3000, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], priors_cxcy))  # (3000, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()
            
            after_iou_box=[]
            after_iou_scores=[]
            after_iou_label=[]
            
            max_scores, best_label = predicted_scores[i].max(dim=1)  # (3000)
            
            # Check for each class
            for c in range(1, n_define_class):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (3000)                
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 3000
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)
            
                max_score_box=[]
                max_score=[]
               
                keep_box=[]
                keep=True
                
                for o_i in range(len(overlap[0])):
                    max_score_box = class_decoded_locs[o_i]
                    max_score = class_scores[o_i]
                    for o_j in range(len(overlap[0])):
                        if(overlap[o_i][o_j]>max_overlap):
                            if(class_scores[o_j]>max_score):
                              keep=False
                              break
                            else:
                               keep=True
                    keep_box.append(keep)
                
                    #print("after_iou_box:",after_iou_box,"after_iou_scores:",after_iou_scores)
              
                # Non-Maximum Suppression (NMS)
                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)
                
                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    #if suppress[box] == 1:
                        
                    if suppress != None:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    
                    suppress = None
                    #suppress[box] = 0

                # Store only unsuppressed boxes for this class
                #if keep_box = Flase, then do suppres=1  to delete the box
                for o_i in range(class_decoded_locs.size(0)):
                    if keep_box[o_i]:
                        suppress[o_i]=suppress[o_i]
                    else:
                        suppress[o_i]=1
                        
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])
               

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size
    
class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
       

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 3000 prior boxes, a tensor of dimensions (N, 3000, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 3000, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
       
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 3000, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 3000)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 3000)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (3000)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (3000)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (3000)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (3000, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 3000)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 3000)
        # So, if predicted_locs has the shape (N, 3000, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 3000)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 3000)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 3000)
        conf_loss_neg[positive_priors] = 0.  # (N, 3000), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 3000), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 3000)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 3000)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss
  