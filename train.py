# ------------------------------------------------------------------------------- #
# Import Libraries
# ------------------------------------------------------------------------------- #

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

# ------------------------------------------------------------------------------- #
# Import Libraries
# ------------------------------------------------------------------------------- #

# Define your transforms for the training, validation, and testing sets
def define_transforms(data_dirs):
    img_datasets, img_dataloaders = [], []
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(data_dirs[0], transform=train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(data_dirs[1], transform=valid_data_transforms)
    test_image_datasets  = datasets.ImageFolder(data_dirs[2], transform=test_data_transforms )
    img_datasets = [train_image_datasets, valid_image_datasets, test_image_datasets]
    
    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64)
    test_dataloader  = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)
    img_dataloaders = [train_dataloader, valid_dataloader, test_dataloader]
    
    # return arrays of datatsets and dataloaders
    return img_datasets, img_dataloaders


#load categories to names file
def load_categories_to_name():
    with open('cat_to_name.json', 'r') as f:
        return json.load(f)

    
# Load a pre-trained network
def load_pretrained_network(args):
    if (type(args.arch) == type(None)) or (args.arch == "vgg16"): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    elif args.arch == "densenet161": 
        model = models.densenet161(pretrained=True)
        model.name = "densenet161"

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    return model


# Define a new, untrainted feed-forward network as a classifier, using ReLU activations and dropout
def feed_forward(model, hidden_unit, arch):
    if arch == 'vgg16':
        in_feature = list(model.children())[1][0].in_features
        model.classifier = nn.Sequential(OrderedDict([                         
                          ('fc1', nn.Linear(in_feature, int(1024 if hidden_unit is None else hidden_unit), bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(int(1024 if hidden_unit is None else hidden_unit), 102, bias=True)), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    elif arch == 'densenet161':
        in_feature = model.classifier.in_features
        model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(int(in_feature), int(1024 if hidden_unit is None else hidden_unit), bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(int(1024 if hidden_unit is None else hidden_unit), 102, bias=True)), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return model


# Define deep learning method and train network
def trainning_network(model, dataloaders, args, optimizer, criterion):
    # Device agnostic code, uses CUDA if it's enabled and user selects it   
    if args.gpu and torch.cuda.is_available():
        device = args.gpu
    else:
        device = "cpu"
    model.to(device)
    
    #number of times model get trained
    epochs = 5
    if args.epochs:
        epochs = args.epochs
        
    # Prints every 30 images out of a predefined batch of images
    print_every = 30 
    steps = 0
    
    # Train the classifier layers using backpropogation using the pre-trained network to get features
    print("Training process starts .....\n")
    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        running_loss = 0
        
        #set to model to train mode
        model.train() 
        
        for inputs, labels in dataloaders[0]:                       
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)   
            
            #unaccumulates the gradients on subsequent backward passes
            optimizer.zero_grad()
            
            #run sequential order as defined in feed_forward             
            outputs = model.forward(inputs)
            
            # Define loss and optimizer
            loss = criterion(outputs, labels)
            
            #back propagation
            loss.backward()
            
            #updates the parameters
            optimizer.step()        
                       
            running_loss += loss.item()
                        
            if steps % print_every == 0:
                #set to evaluation process
                model.eval()
                valid_loss = 0
                accuracy = 0
                
                #disabled gradient calculation in eval process
                with torch.no_grad():
                    for inputs, labels in dataloaders[1]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        log_ps = model.forward(inputs)
                        valid_loss += criterion(log_ps, labels).item()
                        ps = torch.exp(log_ps) 
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = (top_class == labels.view(*top_class.shape))
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders[1]):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders[1]):.3f}")

                train_losses.append(running_loss/len(dataloaders[0]))
                valid_losses.append(valid_loss/len(dataloaders[1]))

                running_loss = 0
                model.train()
    
    
#Do validation on the test dataset    
def test_dataset(model, dataloaders, gpu, criterion):   
    print("Testing process starts .....\n")
    if gpu and torch.cuda.is_available():
        device = gpu
    else:
        device = "cpu"

    test_loss = 0
    accuracy = 0
    model.eval()

    #set to evaluation process (w/o gradient calculatin)
    with torch.no_grad():
        for inputs, labels in dataloaders[2]:
            inputs, labels = inputs.to(device), labels.to(device)
            test_ps = model.forward(inputs)
            test_loss += criterion(test_ps, labels).item()
            ps = torch.exp(test_ps) 
            top_p, top_class = ps.topk(1, dim=1)
            equals = (top_class == labels.view(*top_class.shape))
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Test loss: {test_loss/len(dataloaders[2]):.3f}.. "
              f"Test accuracy: {accuracy/len(dataloaders[2]):.3f}")

        print("\nTesting process ends!")

        
# Parses inputs from the command line
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    # Add checkpoint directory to parser
    parser.add_argument('--data_dir', 
                        type=str, 
                        help='Directory for images.')
    
    # Add architecture selection to parser
    parser.add_argument('--arch', 
                        type=str, 
                        help='Select an architecture from torchvision.models')
        
    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate')
    
    parser.add_argument('--hidden_unit', 
                        type=int, 
                        help='Hidden units for DNN classifier')
    
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training')

    # Add Option to parser
    parser.add_argument('--gpu', 
                        type=str,  
                        help='Use GPU or Cuda for calculations')
    
    # Parse args
    args = parser.parse_args()
    return args


#save model so you can use it later
def save_checkpoint(model, image_datasets, optimizer, epochs, arch, name='checkpoint.pth'):   
    model.class_to_idx = image_datasets.class_to_idx
    if arch == "vgg16":       
        checkpoint = {'input_size': 2208,
                  'output_size': 102,
                  'classifier': model.classifier,
                  'optimizer': optimizer,
                  'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'arch':model.name}
    elif arch == "densenet161":
        checkpoint = {'input_size': 2208,
                      'output_size': 102,
                      'classifier': model.classifier,
                      'optimizer': optimizer,
                      'epochs': epochs,
                      'state_dict': model.state_dict(),
                      'class_to_idx': model.class_to_idx,
                      'arch':model.name}
    torch.save(checkpoint, name)
    
    
#load a checkpoint and rebuild the model so you can come back without having to retrain the network.    
def load_checkpoint(arch, filepath='checkpoint.pth'):
    # Load saved file
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet161':
        #model = torchvision.models.densenet161()
        #self.features = model._modules[‘features’]
        #self.block = self.features._modules[‘denseblock1’]
        model = torchvision.models.densenet161(pretrained=True)
        
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        model.classifier = checkpoint['classifier']
        optimizer = checkpoint['optimizer']
        epochs = checkpoint['epochs']
        model.load_state_dict(checkpoint['state_dict'])
        if arch == "vgg16":
            model.class_to_idx = checkpoint['class_to_idx']
    return model
        
    
def main():
    """Method to train a new network on a data set in data_directory
        Args: 
            None
        Returns: 
            float: mean of the data set
    """
    
    '''Get argument values'''   
    args = arg_parser()

    #default to local flowers directory
    data_dir = "flowers"
    if args.data_dir:
        data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_dirs = [train_dir, valid_dir, test_dir]
        
    #define transform for the training, validation, and testing sets
    datasets, dataloaders = define_transforms(data_dirs)
  
    #load pretrained network
    model = load_pretrained_network(args)
        
    #define untrained feed-forward network
    model = feed_forward(model, args.hidden_unit, args.arch)

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    
    learning_rate = 0.001    
    if args.learning_rate:
        learning_rate = args.learning_rate
        
    epochs = 5
    if args.epochs:
        epochs = args.epochs

    #set optimized algorithm to update the various parameters that can reduce the loss
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    #train the network using backpropagation
    #trainning_network(model, dataloaders, args, optimizer, criterion)
    
    #validate network on the test (untrained) dataset 
    #test_dataset(model, dataloaders, args.gpu, criterion)
    
    #save model for future use
    save_checkpoint(model, datasets[0], optimizer, epochs, ('vgg16' if args.arch is None else args.arch), 'checkpoint_densenet161.pth')    

    
if __name__ == '__main__':
    main()
