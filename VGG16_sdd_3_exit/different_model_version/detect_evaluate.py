from openvino.inference_engine import IECore
import cv2 as cv
from torch.nn import functional as F
import torch
from math import sqrt
import time
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import entropy
import warnings
import os
import psutil

#
sys.path.append("..") 
from datasets import PascalVOCDataset
from utils import *


warnings.filterwarnings("ignore", category=UserWarning)#ignore some unimportant warning messages


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device for tensor calculation

top_k=200
n_classes=21
batch_size =1

exit_threshold1 = 0.8
exit_threshold2 = 0.8
num_exit = 3
min_score=0.2 # minimal score of the box
max_overlap=0.4 #maximum overlap of different box
test_images_number = 10

model_backbone0_xml = "./weight/weight_float/backbone0.xml"
model_backbone0_bin = "./weight/weight_float/backbone0.bin"
model_backbone1_xml = "./weight/weight_float/backbone1.xml"
model_backbone1_bin = "./weight/weight_float/backbone1.bin"
model_backbone2_xml = "./weight/weight_float/backbone2.xml"
model_backbone2_bin = "./weight/weight_float/backbone2.bin"     
model_sub1_xml = "./weight/weight_float/sub1.xml" 
model_sub1_bin = "./weight/weight_float/sub1.bin"
model_sub2_xml = "./weight/weight_float/sub2.xml"
model_sub2_bin = "./weight/weight_float/sub2.bin"

data_folder = './' 
keep_difficult = True
workers = 1

device = "cuda" if torch.cuda.is_available() else "cpu" 


#------------------------label definition-----------------------------------
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

#---------------------------------Load test data--------------------------------
test_dataset = PascalVOCDataset(data_folder,  
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

#-----------------------prior box setting-------------------------------------
def create_prior_boxes():
    """
    Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.

    :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
    """
    fmap_dims = {'conv4_3': 38,
                 'conv7': 19,
                 'conv8_2': 10,
                 'conv9_2': 5,
                 'conv10_2': 3,
                 'conv11_2': 1}

    obj_scales = {'conv4_3': 0.1,
                  'conv7': 0.2,
                  'conv8_2': 0.375,
                  'conv9_2': 0.55,
                  'conv10_2': 0.725,
                  'conv11_2': 0.9}

    aspect_ratios = {'conv4_3': [1., 2., 0.5],
                     'conv7': [1., 2., 3., 0.5, .333],
                     'conv8_2': [1., 2., 3., 0.5, .333],
                     'conv9_2': [1., 2., 3., 0.5, .333],
                     'conv10_2': [1., 2., 0.5],
                     'conv11_2': [1., 2., 0.5]}

    fmaps = list(fmap_dims.keys())

    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                    # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                    # scale of the current feature map and the scale of the next feature map
                    if ratio == 1.:
                        try:
                            additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        # For the last feature map, there is no "next" feature map
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

    prior_boxes = torch.FloatTensor(prior_boxes)  # (8732, 4)
    prior_boxes.clamp_(0, 1)  # (8732, 4)

    return prior_boxes

def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h

#--------------------------IOU setting---------------------------------------
def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def find_jaccard_overlap(set_1, set_2):
  
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

#-------------------------process the output from network-------------------
def detect_objects(predicted_locs, predicted_scores, min_score, max_overlap, top_k):
    """
    Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

    For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

    :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
    :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
    :param min_score: minimum threshold for a box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :return: detections (boxes, labels, and scores), lists of length batch_size
    """
    priors_cxcy = create_prior_boxes()
    #batch_size = batch_size
    n_priors = priors_cxcy.size(0)
    predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)
    
    # Lists to store final predicted boxes, labels, and scores for all images
    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

    for i in range(batch_size):
        # Decode object coordinates from the form we regressed predicted boxes to
        decoded_locs = cxcy_to_xy(
            gcxgcy_to_cxcy(predicted_locs[i], priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

        # Lists to store boxes and scores for this image
        image_boxes = list()
        image_labels = list()
        image_scores = list()
        
        after_iou_box=[]
        after_iou_scores=[]
        after_iou_label=[]
        
        max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)
        
        # Check for each class
        for c in range(1, n_classes):
            # Keep only predicted boxes and scores where scores for this class are above the minimum score
            class_scores = predicted_scores[i][:, c]  # (8732)                
            score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
            n_above_min_score = score_above_min_score.sum().item()
            if n_above_min_score == 0:
                continue
            class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
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
            suppress = torch.zeros((n_above_min_score), dtype=torch.uint8)  # (n_qualified)
            
            # Consider each box in order of decreasing scores
            for box in range(class_decoded_locs.size(0)):
                # If this box is already marked for suppression
                #if suppress[box] == 1:
                    
                if suppress !=None :
                    continue

                # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                # Find such boxes and update suppress indices
                suppress = torch.max(suppress)
                # The max operation retains previously suppressed boxes, like an 'OR' operation

                # Don't suppress this box, even though it has an overlap of 1 with it
                
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
            image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]))
            image_scores.append(class_scores[1 - suppress])
           

        # If no object in any class is found, store a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]))
            image_labels.append(torch.LongTensor([0]))
            image_scores.append(torch.FloatTensor([0.]))

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

#-----------------------------exit strategy-------------------------------
def exit_criterion_top1( location,prediction,exit_threshold): #NOT for batch size > 1
    #evaluate the exit criterion on the result provided
    #return true if it can exit, false if it can't
    location = torch.tensor(location) 
    prediction =  torch.tensor(prediction) 
    
    exit_bool=False
    prediction = F.softmax(prediction, dim=2)
    max_prediction=np.max(np.array((prediction[0][:,1:-1]).cpu()))#要剔除背景一类的预测值
    batch_size = location.size(0)
    for i in range(batch_size):
        if (max_prediction) > (exit_threshold) :
            exit_bool=True
            #print("\n--------------Early Exit----------------\n")
            break
    return exit_bool

def exit_criterion_entropy( location,prediction,exit_threshold): #NOT for batch size > 1
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
        batch_size_ = batch_size
        for i in range(batch_size_):
            if ((entropy_value)) < (exit_threshold) :
                exit_bool=True        
                print("entropy",entropy_value)
                return exit_bool
    return exit_bool

#---------------------------visualize the result----------------------------------
def last_process(det_boxes,det_labels,det_scores):
    
    det_boxes = det_boxes[0].to('cpu')
  
    original_image = Image.open(IMAGE_PATH, mode='r')
    original_image = original_image.convert('RGB')
        
    # --------Transform to original image dimensions--------
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
 
    # --------Decode class integer labels--------
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    
    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        return original_image # Just return original image
    
    print("label:",det_labels,"\nconfidence:",det_scores,"\n")\
        
    # -------Annotate output information on image-------
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)
 
    # draw box and text on image
    for i in range(det_boxes.size(0)):
        
        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  
 
        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], 
                            box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw
    
    
    return annotated_image

#-------------------------------device inference-----------------------------------
def inference_main(images):
    
    result_all=[]
    inference_time = []
    read_network_time = 0
    #---------------define the network--------------------------
    ie_ = IECore()
    device_name = ("MYRIAD" if "MYRIAD" in str(ie_.available_devices) else "CPU")   
    
    #---------------------------read the network-----------------------
    net_ = ie_.read_network(model=model_backbone0_xml, weights= model_backbone0_bin)
    
    input_blob = next(iter(net_.input_info))
    out_blob = next(iter(net_.outputs))
   
    #----------------------------do the inference-------------------------
    net_.batch_size=batch_size
    exec_net = ie_.load_network(network=net_, device_name=device_name)
    
    start_time = time.time()
    result_backbone0= exec_net.infer(inputs={input_blob:[images]})
    infer_backbone0_time = time.time()-start_time
    inference_time.append(infer_backbone0_time)
    
    #---------------------------read the sub network-----------------------  
    net_ = ie_.read_network(model=model_sub1_xml, weights= model_sub1_bin) 
    net_.batch_size=batch_size
    input_dict ={'input':result_backbone0['output']}
    exec_net = ie_.load_network(network=net_, device_name=device_name)
    
    start_time = time.time()
    result= exec_net.infer(inputs=input_dict, )
    infer_sub1_time = time.time()-start_time 
    
    result_all.append(result)
    
  
    #------------------------------decide if need to early exit---------------
    exit_bool = exit_criterion_top1(result['output1'],result['output2'],exit_threshold1)
    exit_point = 0 
    
    if(not exit_bool):
         
        net_ = ie_.read_network(model=model_backbone1_xml, weights= model_backbone1_bin)
        
        net_.batch_size=batch_size
        input_dict ={'input':result_backbone0['output']}
       
        exec_net = ie_.load_network(network=net_, device_name=device_name)
        
        start_time = time.time()
        result_backbone1= exec_net.infer(inputs=input_dict, )
        infer_backbone1_time = time.time()-start_time
       
        #--------------------------sub2 network---------------------------------
      
        net_ = ie_.read_network(model=model_sub2_xml, weights= model_sub2_bin)
      
        net_.batch_size=batch_size
        
        input_dict ={'input1':result_backbone1['output1'],'input2':result_backbone1['output2']}
        exec_net = ie_.load_network(network=net_, device_name=device_name)
        
        start_time = time.time()
        result= exec_net.infer(inputs=input_dict, )
        infer_sub2_time = time.time()-start_time 
        
        inference_time.append(infer_sub2_time+infer_backbone0_time+infer_backbone1_time
                                                          +infer_sub1_time)
        result_all.append(result)
        
        #-------------------------------exit final (whole main backbone)--------
        exit2_bool = exit_criterion_top1(result['output1'],result['output2'],exit_threshold2)
        exit_point = 1
        if(not exit2_bool):
            exit_point = 2
            net_ = ie_.read_network(model=model_backbone2_xml, weights= model_backbone2_bin)
           
            net_.batch_size=batch_size
            input_dict ={'input1':result_backbone1['output1'],'input2':result_backbone1['output2']}
            exec_net = ie_.load_network(network=net_, device_name=device_name)
            
            start_time = time.time()
            result= exec_net.infer(inputs=input_dict, )
            infer_backbone2_time = time.time()-start_time 
            
            result_all.append(result)
            inference_time.append(infer_backbone2_time+infer_backbone0_time+infer_backbone1_time
                                                                  +infer_sub1_time+infer_sub2_time)
   
    return result,exit_point,inference_time
    
     
def evaluate()   :
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()
    
    eixt_1_inference_time = AverageMeter()   
    eixt_2_inference_time = AverageMeter()
    eixt_Final_inference_time = AverageMeter()
    inference_time_total = 0
    exit_1_num=0
    exit_2_num=0
    exit_F_num=0
    
    with torch.no_grad():
        
        for i, (images, boxes, labels, difficulties) in enumerate(test_loader):
            
            images = images.numpy()
            
            result,exit_point,inference_time = inference_main(images)
            
            #Count the number of each exits point do the exit behavior
            if(exit_point==0):
                exit_1_num+=1
            elif exit_point==1:
                exit_2_num+=1
            else:
                exit_F_num+=1
            
            
            #-------------------------process the output information---------
            predicted_scores= result['output2']
            predicted_locs  = result['output1'] 
            
            predicted_locs = torch.tensor(predicted_locs) 
            predicted_scores =  torch.tensor(predicted_scores) 
            
            det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                                    max_overlap=max_overlap,top_k=top_k)
             
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
            
            #mAP calculation
            APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
            
            #different exit point time consume record
            if(exit_point==0): 
                eixt_1_inference_time.update(inference_time[0],1)
            elif(exit_point==1):
                eixt_1_inference_time.update(inference_time[0],1)
                eixt_2_inference_time.update(inference_time[1],1)
            else :
                eixt_1_inference_time.update(inference_time[0],1)
                eixt_2_inference_time.update(inference_time[1],1)
                eixt_Final_inference_time.update(inference_time[2],1)
            
            inference_time_total +=inference_time[-1]
            if((i+1)%10==0):print(i,":",mAP)
            if i>test_images_number:break
           
        #---------------output the result------------------------
        print("----------------------------------------------")        
        print("device name is CPU")
        print("loss function is exit_criterion_top1", "with exit_threshold1:",exit_threshold1,"and exit_threshold2:",exit_threshold2,)
        print("probability 1st early exit:" ,exit_1_num/i*100,"%" ,",with",eixt_1_inference_time.avg,"for each image")
        print("probability 2nd early exit:" ,exit_2_num/i*100,"%",",with",eixt_2_inference_time.avg,"for each image")
        print("probability Final early exit:" ,exit_F_num/i*100,"%",",withA",eixt_Final_inference_time.avg,"for each image")
        print()
        print("mAP :", mAP,",inference total time:",inference_time_total)
        print("----------------------------------------------")        
        
        
if __name__ == '__main__':
    evaluate() 