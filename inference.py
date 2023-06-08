import torch
import numpy as np
from torchvision import transforms
from torch import nn
from transformers import ConvNextConfig, UperNetConfig,ConvNextV2Config,AutoModel, UperNetForSemanticSegmentation, SwinConfig, Swinv2Config
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor
from torchvision.models import segmentation
from torchvision.io import read_image
import os
from PIL import Image
import glob
import albumentations as A
import pandas as pd
import monai

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
from torchmetrics.classification import BinaryJaccardIndex
import sys 
from conv_next import create_model

name = 'convnext'
num_classes = 1 
output_path = r'/kaggle/working'
model = create_model(name , num_classes )
model.load_state_dict(torch.load('./config/convnext_model_v2_epoch.pth' , map_location = 'cpu'))
model.to('cpu')

          
transforms_test = A.Compose([A.Resize(256,256), 
                                 #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                  
                                    ])

def calculate_iou(predicted_mask, ground_truth_mask):
    # Threshold predicted mask to binary values
    predicted_mask = (predicted_mask > 0.5).float()

    # Calculate intersection and union
    intersection = torch.sum(predicted_mask * ground_truth_mask)
    union = torch.sum(predicted_mask) + torch.sum(ground_truth_mask) - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-7)  # Add epsilon to avoid division by zero

    return iou


def call_predict( image ) :



    with torch.no_grad():
        pred = model(image.unsqueeze(0))['logits']
        
    

    predicted_mask = torch.sigmoid(pred)
    
    return predicted_mask 


def preprocess_image( image_path ) : 
    x = Image.open(image_path ).convert('RGB')
    
    transform_images = transforms_test(image = np.array(x) )
        
    return torch.from_numpy(transform_images['image']/255.0).permute(2,0,1).to(torch.float32) 


def run_inference(image_path) :
    print('the image path is ' , image_path)
    x = preprocess_image(image_path) 
    output_image = call_predict(x) 
    return output_image


if __name__ == '__main__' :
    image_path = sys.argv[1]
    final_output = run_inference(image_path)
    print(final_output)
