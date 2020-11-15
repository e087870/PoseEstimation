import os
import re
import sys
sys.path.append('.')
import cv2
from PIL import Image
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='./train/best_pose/best_pose_epoch50.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)


model = get_model('vgg19')     
#model.load_state_dict(torch.load(args.weight))
# original saved file with DataParallel
state_dict = torch.load(args.weight)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

# Motivation for the real-time implementation of pose estimation
# https://realpython.com/face-detection-in-python-using-a-webcam/

# cleanup the image folder
mypath = "./LiveImages"
for root, dirs, files in os.walk(mypath):
    for file in files:
        os.remove(os.path.join(root, file))
        

# webcam image source
video_capture = cv2.VideoCapture(0)
imgnum=0

while True:
    
    ret, frame = video_capture.read()      
           
    shape_dst = np.min(frame.shape[0:2])   
       
    with torch.no_grad():
        paf, heatmap, im_scale = get_outputs(frame, model,  'rtpose')              
    
    humans = paf_to_pose_cpp(heatmap, paf, cfg)
            
    out = draw_humans(frame, humans)      
        
    cv2.imshow('Video', out)    
    
    # skip the first image and write the rest of the stream
    if (imgnum > 0):
        cv2.imwrite('./LiveImages/image' + str(imgnum) + '.png',out) 

    # break loop with key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    imgnum+=1
    

# create the video
print ('Creating video from the images')

img_array = []

for idx in range(1,imgnum):    
    img = cv2.imread('./LiveImages/image' + str(idx) + '.png')
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img) 
 
out = cv2.VideoWriter('./video/pose_estimation.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
    
out.release()   
