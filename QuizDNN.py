
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import torchvision
from torchsummary import summary
from tqdm import tqdm

from albumentations import  ( 
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose , Normalize ,ToFloat, Cutout
)

import cv2

import numpy as np

from albumentations.pytorch import  ToTensor 

def QDNN():
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          # Input Block
          self.convblock1 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
              nn.BatchNorm2d(8),
              nn.ReLU()
            
          ) 

          # CONVOLUTION BLOCK 1
          self.convblock2 = nn.Sequential(
              nn.Conv2d(in_channels=11, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
              nn.BatchNorm2d(16),
              nn.ReLU()
            
          )
          

          # TRANSITION BLOCK 1
          self.pool1 = nn.MaxPool2d(2, 2)


          # Input Block
          self.convblock3 = nn.Sequential(
              nn.Conv2d(in_channels=27, out_channels=38, kernel_size=(3, 3), padding=1, bias=False),
              nn.BatchNorm2d(38),
              nn.ReLU()
            
          ) 

          # CONVOLUTION BLOCK 1
          self.convblock4 = nn.Sequential(
              nn.Conv2d(in_channels=65, out_channels=76, kernel_size=(3, 3), padding=1, bias=False),
              nn.BatchNorm2d(76),
              nn.ReLU()
            
          )
          
          # CONVOLUTION BLOCK 1
          self.convblock5 = nn.Sequential(
              nn.Conv2d(in_channels=141, out_channels=152, kernel_size=(3, 3), padding=1, bias=False),
              nn.BatchNorm2d(152),
              nn.ReLU()
            
          )

          
          # CONVOLUTION BLOCK 1
          self.convblock6 = nn.Sequential(
              nn.Conv2d(in_channels=266, out_channels=274, kernel_size=(3, 3), padding=1, bias=False),
              nn.BatchNorm2d(274),
              nn.ReLU()
            
          )

          # CONVOLUTION BLOCK 1
          self.convblock7 = nn.Sequential(
              nn.Conv2d(in_channels=540, out_channels=548, kernel_size=(3, 3), padding=1, bias=False),
              nn.BatchNorm2d(548),
              nn.ReLU()
            
          )

          # CONVOLUTION BLOCK 1
          self.convblock8 = nn.Sequential(
              nn.Conv2d(in_channels=1088, out_channels=1024, kernel_size=(3, 3), padding=1, bias=False),
              nn.BatchNorm2d(1024),
              nn.ReLU()
            
          )
          
          
          #GAP kernel
          self.gap = nn.Sequential(
              nn.AvgPool2d(kernel_size=7)
          ) 
          #Fully connected layers
          self.fc1 = nn.Sequential(
                    nn.Linear(1024, 10)
                    )

        
      def forward(self, x):

          x2 = self.convblock1(x) 

          x3 = self.convblock2(torch.cat([x,x2],dim=1)) 

          x4 = self.pool1(torch.cat([x,x2,x3],dim=1))
          
          x5 = self.convblock3(x4)

          x6 = self.convblock4(torch.cat([x4,x5],dim=1)) 

          x7 = self.convblock5(torch.cat([x4,x5,x6],dim=1))

          x8 = self.pool1(torch.cat([x5,x6,x7],dim=1))

          x9 = self.convblock6(x8)

          x10 = self.convblock7(torch.cat([x8,x9],dim=1)) 

          x11 = self.convblock8(torch.cat([x8,x9,x10],dim=1))

          
          x12 = self.gap(x11)

          # print(x12.shape)

          x13 = x12.view(-1, 1024)

          x14 = self.fc1(x13)


          return F.log_softmax(x14, dim=-1)
  return Net