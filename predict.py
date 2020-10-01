### Imports

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json



###Parser

parser = argparse.ArgumentParser(description = "Predicttion Script Parser")

parser.add_argument('image_dir', help = 'Provide path to image. Mandatory argument', type = str)
parser.add_argument('load_dir', help = 'Provide path to checkpoint. Mandatory argument', type = str)
parser.add_argument('--top_k', help = 'Top K most likely classes. default value is 3', type = int)
parser.add_argument('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. default value is cat_to_name.json', type = str)
parser.add_argument('--GPU', help = "Option to use GPU. Enter True/False. default value is False", type = str)

#Setting up parser
args = parser.parse_args()



###Setup Variables

image_dir = args.image_dir

checkpoint = args.load_dir

if args.top_k:
    top_k = args.top_k
else:
    top_k = 3
    
if args.category_names:
    category_names = args.category_names
    with open(category_names, 'r') as f:
        category_names = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        category_names = json.load(f) 

if args.GPU == 'True':
    user_device = 'cuda'
else:
    user_device = 'cpu'
    
    
    
###Build Model from checkpoint

def checkpoint_load(filepath):
    
    if user_device == 'cpu':
        checkpoint = torch.load(filepath, map_location=lambda storage, loc:storage)
    else:
        checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'VGG13':
        model = models.vgg13(pretrained = True)
        print('The model architecture is VGG13')
        print('\n')
    else:
        model = models.vgg11(pretrained = True)
        print('The model architecture is VGG11')
        print('\n')
    
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model



### Process the Image

def process_image(image):
    #Open Image
    img_pil = Image.open(image)
   
    #Reshape image to the right format
    right_format = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = right_format(img_pil)
    return img_tensor




### Function to predict the class from image file

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Loading model - using .cpu() for working with CPUs
    loaded_model = checkpoint_load(model).cpu()
    
    # Pre-processing image
    img_tensor = process_image(image_path)
    
    #Correct Dimensions
    img_add_dim = img_tensor.unsqueeze_(0)

    # Running model
    loaded_model.eval()
    with torch.no_grad():
        output = loaded_model.forward(img_add_dim)

    # Calculating probabilities
    probs = torch.exp(output)
    probs_topk = probs.topk(topk)[0]
    index_topk = probs.topk(topk)[1]

    
    # Converting probabilities and outputs to lists
    probs_topk_list = np.array(probs_topk)[0]
    index_topk_list = np.array(index_topk)[0]
    
    # Loading index and class mapping
    class_to_idx = loaded_model.class_to_idx
    
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_topk_list = []
    for index in index_topk_list:
        classes_topk_list += [indx_to_class[index]]
        
    return probs_topk_list, classes_topk_list




### Executing Predict Function

model_path = checkpoint
image_path = image_dir

probs,classes = predict(image_path, model_path, topk=top_k)

names = []
for i in classes:
    names += [category_names[i]]
    
print('The predicted labels for this flower:','\n', names)
print('The predicted probability of each label:','\n', probs)