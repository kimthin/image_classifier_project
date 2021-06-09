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
from train import load_pretrained_network


def load_categories_to_name():
    with open('cat_to_name.json', 'r') as f:
        return json.load(f)
    
def save_checkpoint():
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'input_size': 2208,
                  'output_size': 102,
                  'classifier': classifier,
                  'optimizer': optimizer,
                  'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,}
    torch.save(checkpoint, 'checkpoint.pth')
    
def load_checkpoint(model, filepath):
    checkpoint = torch.load(filepath)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        model.classifier = checkpoint['classifier']
        optimizer = checkpoint['optimizer']
        epochs = checkpoint['epochs']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path) 
    width, height = image.size 
    small_size_to = 256
    if width > height: 
        image.resize((width, small_size_to))
    else: 
        image.resize((small_size_to,height))
            
    width, height = image.size 
    reduce_size = 224
    left = (width - reduce_size)/2 
    right = left + reduce_size
    top = (height - reduce_size)/2
    bottom = top + reduce_size
    image = image.crop ((left, top, right, bottom))
    
    mean = np.array ([0.485, 0.456, 0.406]) 
    std = np.array ([0.229, 0.224, 0.225])
    np_image = np.array(image)/255
    np_image -= mean
    np_image /= std
    
    np_image= np_image.transpose ((2,0,1))
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict_image(image_path, model, topk=5, gpu="cpu"):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file       
    if gpu == "cuda":
        model.cuda()
    else:
        model.cpu()
        
    model.eval()
    
    np_image = process_image(image_path)
    torch_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    torch_image = torch_image.unsqueeze(dim = 0)
    
    with torch.no_grad():
        if gpu == "cuda":
            torch_image = torch_image.cuda()
        else:
            torch_image = torch_image.cpu()
        
        out = model.forward(torch_image)
        ps = torch.exp(out) 
        
        
        topk_prob = torch.topk(ps, int(topk))[0].tolist()[0]            # probability
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
                        help='Point to checkpoint file as str.')
    
    # Add --arch to parser
    parser.add_argument('--arch', 
                        type=str, 
                        help='Architecture used in training model')
    
    # Parse args
    args = parser.parse_args()
    return args

    
def main():
    args = arg_parser()
    print(args)
    
    image_path = "flowers/train/1/image_06742.jpg"
    if args.image:
        image_path = args.image

    #arch = "vgg16"
    #if args.arch:
    #    arch = args.arch
    #print(arch)
        
    #topk = 5
    #if args.topk:
    #    topk = args.topk
    
    #checkpoint = "checkpoint.pth"
    #if args.checkpoint:
    #    checkpoint = args.checkpoint
                
    #arch = "vgg16"
    #if args.arch:
    #    arch = args.arch
       
    model = load_pretrained_network(args.arch)

    #if model.arch is "vgg16"
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint)
    #else:
    #    save_checkpoint()
    topk = 5  
    if args.topk:
        topk = args.topk
        
    probs, classes = predict_image(image_path, model, topk)
    print(probs)
    print(classes)

        
if __name__ == '__main__':
    main()        