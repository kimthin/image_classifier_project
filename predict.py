import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from PIL import Image
from torch.autograd import Variable
import seaborn as sns
import torchvision
import sys
import argparse
import json
import train
from train import load_checkpoint
from train import define_transforms

#load a file that maps category label to category name
def load_categories_to_name():
    with open('cat_to_name.json', 'r') as f:
        return json.load(f)
 

#Process a PIL image for use in a PyTorch model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #read image from file
    image = Image.open(image_path) 
    width, height = image.size 
    small_size_to = 256
    
    #resize image shorter size to 256 pixels, keeping the aspect ratio
    if width > height: 
        image.resize((width, small_size_to))
    else: 
        image.resize((small_size_to,height))

    #crop out the center 224x224 portion of the image
    width, height = image.size 
    reduce_size = 224
    left = (width - reduce_size)/2 
    right = left + reduce_size
    top = (height - reduce_size)/2
    bottom = top + reduce_size   
    image = image.crop ((left, top, right, bottom))
    
    mean = np.array ([0.485, 0.456, 0.406]) 
    std = np.array ([0.229, 0.224, 0.225])
    
    #convert color channels of images as integers 0-255 to floats 0-1 which the model expected
    #first convert image to numpy array then scale it down to values from 0-1.
    #then get the mean color by substrating mean array and divided by standard deviation array
    np_image = np.array(image)/255
    np_image -= mean
    np_image /= std
    
    #pytorch expects color channel to be first dimension, but it's the third dimension in the PIL image and Numpy array
    #hence reorder dimensions using ndarray.transpose
    np_image= np_image.transpose ((2,0,1))
    return np_image

# Define a new, untrainted feed-forward network as a classifier, using ReLU activations and dropout
def feed_forward(model, arch, hidden_unit=1024):
    if arch == 'vgg16':
        in_feature = list(model.children())[1][0].in_features
        model.classifier = nn.Sequential(OrderedDict([                         
                          ('fc1', nn.Linear(int(in_feature), hidden_unit, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_unit, 102, bias=True)), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    elif arch == 'densenet161':
        in_feature = list(model.children())[0][:-2]
        model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', 
                           nn.Linear(in_feature, 
                           hidden_unit, 
                           bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_unit, 102, bias=True)), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return model


#predict the class from an image file
def predict_image(args, image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
       
    if args.gpu and torch.cuda.is_available():
        device = args.gpu
    else:
         device = args.gpu
    model.to(device)

    model.eval()
    
    np_image = process_image(image_path)
    torch_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    torch_image = torch_image.unsqueeze(dim = 0)
    
    with torch.no_grad():
        if args.gpu == "cuda":
            torch_image = torch_image.cuda()
            model.to("cuda")
        else:
            torch_image = torch_image.cpu()
            model.to("cpu")
        
        out = model.forward(torch_image)
        ps = torch.exp(out) 
       
        topk = 5
        if args.topk: 
            topk = args.topk          
        
        topk_prob = torch.topk(ps, topk)[0].tolist()[0]            # probability
        indeces = torch.topk(ps, topk)[1].tolist()[0]              # index

        key_value = {val: key for key, val in model.class_to_idx.items()}

        topk_classes = [key_value[item] for item in indeces]
        topk_classes = np.array(topk_classes)
    
        return topk_prob, topk_classes

# Parses inputs from the command line
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    # Add image to be predicted to parser
    parser.add_argument('--image', 
                        type=str,  
                        help='Image file to be predicted by network',
                        required=True)
    
    parser.add_argument('--hidden_unit', 
                        type=str,  
                        help='Number of units in hidden layer')
        
    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        type=str,  
                        help='Use GPU or Cuda for calculations')
    
    # Add GPU Option to parser
    parser.add_argument('--category_names', 
                        type=str,  
                        help='Json file',
                        required=True)
    
    # Add topk to parser
    parser.add_argument('--topk', 
                        type=int,  
                        help='top numbers of probabilities')
    
    # Load checkpoint created by train.py
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.',
                        required=True)
    
    # Add --arch to parser
    parser.add_argument('--arch', 
                        type=str, 
                        help='Architecture used in training model')
    
    # Parse args
    args = parser.parse_args()
    return args

    
def main():
    args = arg_parser()              
    
    load_categories_to_name()
    
    arch = 'vgg16' if args.arch is None else args.arch
    model = load_checkpoint(arch, args.checkpoint)
    #print(model)
      
    model = feed_forward(model, arch, int(1024 if args.hidden_unit is None else args.hidden_unit))
        
    probs, classes = predict_image(args, args.image, model)
    print(probs)
    print(classes)

        
if __name__ == '__main__':
    main()        