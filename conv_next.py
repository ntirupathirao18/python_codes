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





def create_txtfile_fromcsv( path, output_path , out_name) :
    df = pd.read_csv(path)
    with open(os.path.join(output_path, out_name+'.txt'), 'w') as f:
        for line in list(df['new_filename']):
            f.write(f"{line}\n")




def create_model(name , num_classes , test_sample = True ) :

  if name == 'convnext' :
    backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
  if name == 'convnextv2' :  
    backbone_config = ConvNextV2Config(out_features=["stage1", "stage2", "stage3", "stage4"])
  if name == 'swin' :
    backbone_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
    

  if name == 'internimage'  : 
    model = AutoModel.from_pretrained('OpenGVLab/internimage_t_1k_224' ,trust_remote_code = True ,
                                     
                                     )
    config = UperNetConfig()
    segmentation_model = UperNetForSemanticSegmentation(config)
    #segmentation_model.encoder = model.levels
    segmentation_model.backbone = model
    del segmentation_model.backbone.model.avgpool
    del segmentation_model.backbone.model.head
    
  else :
      config = UperNetConfig(backbone_config=backbone_config)
      #segmentation_model = UperNetForSemanticSegmentation.from_pretrained('openmmlab/upernet-convnext-tiny')
      segmentation_model = UperNetForSemanticSegmentation(config)

  
    
  

  in_channels = segmentation_model.decode_head.classifier.in_channels
  print(in_channels)
  # Number of output channels equal to number of classes
  
  output_channels = num_classes
  segmentation_model.decode_head.classifier = nn.Conv2d(512, output_channels, kernel_size=(1, 1), stride=(1, 1))
  segmentation_model.auxiliary_head.classifier = nn.Conv2d(256, output_channels, kernel_size=(1, 1), stride=(1, 1))

  if test_sample :
  # Verify the model output
      with torch.no_grad():
        seg_output = segmentation_model(torch.zeros(size= (4, 3, 224, 224)))

      print(seg_output.logits.shape)
  return segmentation_model 

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SegDataset(Dataset):
    
    def __init__(self, parentDir, imageDir, maskDir):
        self.imageList = glob.glob(parentDir+'/'+imageDir+'/*')
        self.imageList.sort()
        self.maskList = glob.glob(parentDir+'/'+maskDir+'/*')
        self.maskList.sort()
    def __getitem__(self, index):
        
        preprocess = transforms.Compose([
                                    transforms.Resize((256,256), 2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        X = Image.open(self.imageList[index]).convert('RGB')
        X = preprocess(X)
        
        trfresize = transforms.Resize((256, 256), 2)
        trftensor = transforms.ToTensor()
        
        yimg = Image.open(self.maskList[index]).convert('L')
        y1 = trftensor(trfresize(yimg))
        y1 = y1.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        y = torch.cat([y2, y1], dim=0)
        
        return X, y1.float()
            
    def __len__(self):
        return len(self.imageList)
    
class SegDataset_text(Dataset):
    
    def __init__(self, parentDir, imageDir, maskDir, txt_path , transforms_data , img_ext= '.jpg' , mask_ext = '.png'):
        with open(txt_path) as f:
            image_ids = f.readlines()
        #self.imageList = glob.glob(parentDir+'/'+imageDir+'/*')
        self.imageList = [os.path.join(parentDir , imageDir, indid.strip()+img_ext)for indid in image_ids ]
        self.imageList.sort()
        #self.maskList = glob.glob(parentDir+'/'+maskDir+'/*')
        self.maskList = [os.path.join(parentDir , maskDir, indid.strip()+mask_ext)for indid in image_ids ]
        self.maskList.sort()
        self.transforms_data = transforms_data
    def __getitem__(self, index):
        
#         preprocess = transforms.Compose([
#                                     transforms.Resize((256,256), 2),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
#         X = Image.open(self.imageList[index]).convert('RGB')
#         X = preprocess(X)
        
#         trfresize = transforms.Resize((256, 256), 2)
#         trftensor = transforms.ToTensor()
        
#         yimg = Image.open(self.maskList[index]).convert('L')
#         y1 = trftensor(trfresize(yimg))
#         y1 = y1.type(torch.BoolTensor)
#         y2 = torch.bitwise_not(y1)
#         y = torch.cat([y2, y1], dim=0)
        
#         return X, y1.float()
        
        x = Image.open(self.imageList[index]).convert('RGB')
        yimg = Image.open(self.maskList[index]).convert('RGB')
        transform_images = self.transforms_data(image = np.array(x), mask = np.array(yimg))
        y1 = Image.fromarray(transform_images['mask']).convert('L')
        
        
#         y2 = torch.bitwise_not(y1)
#         y = torch.cat([y2, y1], dim=0)
        
        return torch.from_numpy(transform_images['image']/255.0).permute(2,0,1).to(torch.float32), torch.unsqueeze(torch.from_numpy(np.array(y1, dtype= 'bool')).float(),0).to(torch.float32)
        #return torch.from_numpy(transform_images['image']).permute(2,0,1).to(torch.float32), torch.unsqueeze(torch.from_numpy(np.array(y1, dtype= 'bool')).float(),0).to(torch.float32)
    def __len__(self):
        return len(self.imageList)
    
def train_model(model , no_of_epochs, check_point_dir , data_loader , name ) :
  # Define the loss function
  criterion = nn.BCEWithLogitsLoss()
  #criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
  # Define the optimizer
  optimizer = optim.Adam(model.parameters(), lr=0.0001)

  # Move the model to the device
  model = model.to(device)

  # Training loop
  epoch_loss_list = [1.0]

  for epoch in range(no_of_epochs):
      model.train()
      running_loss = 0.0

      for images, masks in data_loader:

          # Move images and masks to the device
          images = images.to(device)
          masks = masks.to(device)

          # Zero the gradients
          optimizer.zero_grad()

          # Forward pass
          outputs = model(images)['logits']

          #print(f"output shape: {outputs.shape}")

          # Compute the loss
          loss = criterion(outputs, masks)

          # Backward pass
          loss.backward()

          # Update the weights
          optimizer.step()

          # Update the running loss
          running_loss += loss.item() * images.size(0)
          """
          fig, axes = plt.subplots()
          axes.imshow(images[1].cpu().detach().numpy())
          show_mask(outputs[1].cpu().detach().numpy(), axes)
          
          print(f"image : {images}")
          print(f"mask : {masks}")
          print(f"outputs : {outputs}")
          """
          
          #print(loss.item())
      # Print the epoch loss
      epoch_loss = running_loss / len(data_loader)
      print(f"Epoch {epoch+1}/{no_of_epochs}, Loss: {epoch_loss:.4f}")

      if epoch_loss < min(epoch_loss_list)  :
        checkpoint_name = os.path.join(check_point_dir, f"{name}_model_epoch.pth") 
        torch.save(model.state_dict(), checkpoint_name)

      epoch_loss_list.append(epoch_loss)

      # if (epoch + 1) % 20 == 0:
      #     checkpoint_name = os.path.join(check_point_dir, f"{name}_model_epoch_{epoch}.pth") 
          
      #     # Save the model parameters
      #     torch.save(model.state_dict(), checkpoint_name)

  return model


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def calculate_iou(predicted_mask, ground_truth_mask):
    # Threshold predicted mask to binary values
    predicted_mask = (predicted_mask > 0.5).float()

    # Calculate intersection and union
    intersection = torch.sum(predicted_mask * ground_truth_mask)
    union = torch.sum(predicted_mask) + torch.sum(ground_truth_mask) - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-7)  # Add epsilon to avoid division by zero

    return iou


def calculate_iou_v2(predicted_mask, ground_truth_mask , metric ): 
    return metric(predicted_mask, ground_truth_mask)

def train_validate_model(model , no_of_epochs, check_point_dir , train_data_loader , name , validation_data_loader) :
  # Define the loss function
    criterion = nn.BCEWithLogitsLoss()
    criterion = DiceBCELoss()
    criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
  # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

  # Move the model to the device
    model = model.to(device)
    metric_v2 = BinaryJaccardIndex().to(device)


  # Training loop
    epoch_loss_list = [1.0]

    for epoch in range(no_of_epochs):
        model.train()
        running_loss = 0.0
        print('epoch'+str(epoch) , end = '')

        for images, masks in tqdm(train_data_loader):

              # Move images and masks to the device
              images = images.to(device)
              masks = masks.to(device)

              # Zero the gradients
              optimizer.zero_grad()

              # Forward pass
              outputs = model(images)['logits']

              #print(f"output shape: {outputs.shape}")

              # Compute the loss
            
              loss = criterion(outputs, masks)

              # Backward pass
              loss.backward()

              # Update the weights
              optimizer.step()

              # Update the running loss
              running_loss += loss.item() * images.size(0)
              """
              fig, axes = plt.subplots()
              axes.imshow(images[1].cpu().detach().numpy())
              show_mask(outputs[1].cpu().detach().numpy(), axes)

              print(f"image : {images}")
              print(f"mask : {masks}")
              print(f"outputs : {outputs}")
              """

              #print(loss.item())
          # Print the epoch loss
        epoch_loss = running_loss / len(train_data_loader)

        if epoch_loss < min(epoch_loss_list)  :
            checkpoint_name = os.path.join(check_point_dir, f"{name}_model_v2_epoch.pth") 
            torch.save(model.state_dict(), checkpoint_name)

        epoch_loss_list.append(epoch_loss)

      ###########validation score with iou metric 
        model.eval()
        validation_metric = 0.0
        validation_metric_v2 = 0.0
    

        for images, masks in tqdm(validation_data_loader):

          # Move images and masks to the device
            images = images.to(device)
            masks = masks.to(device)
            target = model(images)['logits']
            predicted_mask = torch.sigmoid(target)

            metric = calculate_iou(predicted_mask,masks)
            validation_metric += metric.item() * images.size(0)
            #validation_metric = 0.0
            metric = calculate_iou_v2(predicted_mask,masks, metric_v2)
            validation_metric_v2 += metric.item() * images.size(0)


        total_metric_value = validation_metric / len(validation_data_loader)
        total_metric_value_v2 = validation_metric_v2 / len(validation_data_loader)

        print(f"Epoch {epoch+1}/{no_of_epochs}, Loss: {epoch_loss:.4f} , metric_IOU : {total_metric_value :.4f}, metric_IOU_v2 : {total_metric_value_v2 :.4f}" )

      # if (epoch + 1) % 20 == 0:
      #     checkpoint_name = os.path.join(check_point_dir, f"{name}_model_epoch_{epoch}.pth") 
          
      #     # Save the model parameters
      #     torch.save(model.state_dict(), checkpoint_name)

    return model



# train_ratio = 0.7 
# test_ratio = 0.3 
# version = 2
# import os 
# dataset_src_path = r'/kaggle/input/leaf-disease-segmentation-dataset'
# txt_src_path = r'/kaggle/input/leafdisease-image-id-split'
# if version == 1 :
# # Save the final trained model
#   dataset = SegDataset(parentDir= os.path.join(dataset_src_path, 'data/data'),imageDir= "images", maskDir= "masks")
#   # Calculate the sizes of training and testing sets
#   train_size = int(train_ratio * len(dataset))
#   test_size = len(dataset) - train_size

#   # Randomly split the dataset into training and testing sets
#   train_set, test_set = random_split(dataset, [train_size, test_size])
# if version == 2 : 
#     transforms_training = A.Compose([
#                                     A.Resize(256,256),
#                                     A.HorizontalFlip(p=0.5) , 
#                                     A.RandomBrightnessContrast(p=0.2),
#                                     A.RandomRotate90(),
#                                     #A.Affine(scale=0.95, rotate = (0,30)),             
#                                      #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      
#                                   ])
          
#     transforms_test = A.Compose([A.Resize(256,256), 
#                                  #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                  
#                                     ])
        
# #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     train_set = SegDataset_text(parentDir= os.path.join(dataset_src_path, 'data/data'),imageDir= "images", maskDir= "masks" , 
#                     txt_path = os.path.join(txt_src_path, 'train.txt'), 
#                                transforms_data = transforms_training)
#     test_set = SegDataset_text(parentDir=  os.path.join(dataset_src_path, 'data/data'),imageDir= "images", maskDir= "masks" , 
#                     txt_path =os.path.join(txt_src_path, 'val.txt'), 
#                               transforms_data = transforms_test)


# # Create data loaders for training and testing sets
# train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_set, batch_size=1, shuffle=False)




# name = 'convnext'
# num_classes = 1 
# output_path = r'/kaggle/working'
# model = create_model(name , num_classes ) 
# #model.load_state_dict(torch.load(r'/kaggle/working/convnext_model_v2_epoch.pth'))
# # model = train_model_(model , no_of_epochs = 700, check_point_dir = '/content/drive/MyDrive/model' ,  data_loader = train_loader , name = name) 
# model = train_validate_model(model , no_of_epochs = 700, check_point_dir = output_path ,  
#                              train_data_loader = train_loader , name = name , 
#                              validation_data_loader = test_loader) 

# torch.save(model.state_dict(), name+"_leaf_Disease.pth")