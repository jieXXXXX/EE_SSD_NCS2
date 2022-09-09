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

import box_utils
#from vision.ssd.ssd import SSD, GraphPath

#from vision.ssd.config import mobilenetv1_ssd_config as config
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #

n_define_class = 21


# Modified from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.
# In this version, Relu6 is replaced with Relu to make it ONNX compatible.
# BatchNorm Layer is optional to make it easy do batch norm confusion.

class Empty(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self):
        super(Empty, self).__init__()

        # Standard convolutional layers in VGG16
        #intput size # (N, 512, 19, 19)
        # Replacements for FC6 and FC7 in VGG16

        # Load pretrained layers
        #self.load_pretrained_layers()

    def forward(self):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
        # Lower-level feature maps
        return 

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

        #self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = x.mean(3).mean(2)
        #x = self.classifier(x)
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
    """
    VGG base convolutions to produce lower-level feature maps.
    """

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
        #nn.Sequential(nn.Conv2d(576 ,last_channel,kernel_size = 3, stride=2, padding=1),
         #                                nn.ReLU6(inplace=True))
        
        self.features = (conv_1x1_bn(576, last_channel,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        #self.features = (conv_1x1_bn(160, last_channel,
         #                                use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
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
               
        # Load pretrained layers
       

    def forward(self,x):
        
        confidences=[]
        locations = []
        header_index=0
        
        y_14 = self.conv_exit_1(x)
        x = self.conv_exit_2(y_14)
        x = self.features(x)
        #x = x.mean(3).mean(2)
        #x = self.classifier(x)
        
        
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
    """
    VGG base convolutions to produce lower-level feature maps.
    """

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
               
        # Load pretrained layers
       

    def forward(self,x,y_14):
        
        confidences=[]
        locations = []
        header_index=0
        confidence_14, location_14 = self.compute_header(header_index, y_14)   
        confidences.append(confidence_14)
        locations.append(location_14)
        header_index+=1
        
        x = self.features(x)
        #x = x.mean(3).mean(2)
        #x = self.classifier(x)
        
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
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self, n_class=n_define_class):
        super(Mobilenetv2_exixFinal, self).__init__()
        
        self.num_classes = n_class
    
        self._initialize_weights()
        
        self.extras = Extras()()
        
        self.classification_headers = Classification_headers()(width_mult = 1.0,num_classes =self.num_classes )
        
        self.regression_headers = Regression_headers()(width_mult = 1.0)
               
        # Load pretrained layers
       

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


class SSD_mobilenetv2_backbone1(nn.Module):
    def __init__(self, num_classes: int,exit_threshold=0.5, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD_mobilenetv2_backbone1, self).__init__()

        self.num_classes = num_classes
       
        self.backbone = nn.ModuleList()
        self._build_backbone()
        
     
   
        # register layers in source_layer_indexes by adding them to a module list
     
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
        bb = self.backbone[0]
        
        x = bb(x)
    
        return x
   
 
    def _build_backbone(self):
       
        strt_bl = Mobilenetv2_Backbone1()#NOTE was padding 3
        
        self.backbone.append(strt_bl)
  

class SSD_mobilenetv2_backbone2(nn.Module):
    def __init__(self, num_classes: int,exit_threshold=0.5, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD_mobilenetv2_backbone2, self).__init__()

        self.num_classes = num_classes
        

        self.backbone = nn.ModuleList()
        self._build_backbone()
        
        self.source_layer_indexes =  source_layer_indexes = [GraphPath(0, 'conv', 3)
                                                             , 19,
                                                             ]
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)]) 
   
        # register layers in source_layer_indexes by adding them to a module list
     
    def forward(self, x):
        
    
        bb=self.backbone[1]
       
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
            
            #confidence_14,location_14 = self.compute_header(header_index, y)
            #header_index += 1
            #confidences.append(confidence)
            #locations.append(location)
            
            for layer in base_net[start_layer_index:]:
                x = layer(x)
            
        return x,y_14
   
 
    def _build_backbone(self):
       
        strt_bl =Empty()#NOTE was padding 3
        
        self.backbone.append(strt_bl)
        
        bb_layers = []
       
        bb_layers.append(Mobilenetv2_Backbone2())
    
        self.backbone.append(*bb_layers)
    

class SSD_mobilenetv2_backbone3(nn.Module):
    def __init__(self, num_classes: int,exit_threshold=0.5, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD_mobilenetv2_backbone3, self).__init__()

        self.num_classes = num_classes
        

        self.backbone = nn.ModuleList()
        self.exits = nn.ModuleList()  
        self._build_backbone()
        self._build_exits()
        
  
   
        # register layers in source_layer_indexes by adding them to a module list
     
    def forward(self, x,y_14):
        
    
        bb=self.backbone[2]
        ee= self.exits[2]
       
        if("Mobilenetv2_Backbone3") in str(bb):
            base_net = bb.features
            for layer in base_net:
                x = layer(x)
        confidence,location = ee(x,y_14)
            
        return confidence,location
 
    def _build_backbone(self):
       
        strt_bl =Empty()#NOTE was padding 3
        
        self.backbone.append(strt_bl)
        
        bb_layers = []
       
        bb_layers.append(Empty())
    
        self.backbone.append(*bb_layers)
        
        
        bb_layers2 = []
       
        bb_layers2.append(Mobilenetv2_Backbone3())
    
        self.backbone.append(*bb_layers2)
        
    def _build_exits(self):
        #adding early exits/branches
        #early exit 1
        ee1 = Empty()#
        self.exits.append(ee1)
        
        ee2 = Empty()#
        self.exits.append(ee2)
        #final exit
        eeF = Mobilenetv2_exixFinal()
        self.exits.append(eeF)


        
class SSD_mobilenetv2_exit1(nn.Module):
    def __init__(self, num_classes: int,exit_threshold=0.5, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD_mobilenetv2_exit1, self).__init__()

        self.num_classes = num_classes
     

        self.exits = nn.ModuleList()  
        self._build_exits()
        
     
    def forward(self, x):
        
    
    
        ee = self.exits[0]
       
        confidence,location = ee(x)
            
              
        return confidence, location
                
        
   
    def _build_exits(self):
        #adding early exits/branches
        #early exit 1
        ee1 = Mobilenetv2_exit1()#
        self.exits.append(ee1)
        #final exit
 
        
class SSD_mobilenetv2_exit2(nn.Module):
    def __init__(self, num_classes: int,exit_threshold=0.5, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD_mobilenetv2_exit2, self).__init__()

        self.num_classes = num_classes
     

        self.exits = nn.ModuleList()  
        self._build_exits()
        
     
    def forward(self, x,y_14):
        
    
    
        ee = self.exits[1]
       
        confidence,location = ee(x,y_14)
            
              
        return confidence, location
                
        
   
    def _build_exits(self):
        #adding early exits/branches
        #early exit 1
        
        ee1 = Empty()#
        self.exits.append(ee1)
        ee2 = Mobilenetv2_exit2()#
        self.exits.append(ee2)
   
   
    
 