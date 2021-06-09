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

def define_transforms(data_dirs):
    # TODO: Define your transforms for the training, validation, and testing sets
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

    # TODO: Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(data_dirs[0], transform=train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(data_dirs[1], transform=valid_data_transforms)
    test_image_datasets  = datasets.ImageFolder(data_dirs[2], transform=test_data_transforms )
    img_datasets = [train_image_datasets, valid_image_datasets, test_image_datasets]
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64)
    test_dataloader  = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)
    img_dataloaders = [train_dataloader, valid_dataloader, test_dataloader]
    return img_datasets, img_dataloaders

def load_categories_to_name():
    with open('cat_to_name.json', 'r') as f:
        return json.load(f)

# Load a pre-trained network
def load_pretrained_network(arch = "vgg16"):
    #model = models.vgg16(pretrained=True)
    if type(arch) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        model = getattr(models, arch)(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    return model

# Define a new, untrainted feed-forward network as a classifier, using ReLU activations and dropout
def feed_forward(model, hidden_unit):
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1024, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(1024, 102, bias=True)), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    return model

def trainning_network(model, dataloaders, args, optimizer, criterion):
    # Define deep learning method
    device = "cpu"
    if args.gpu:
        device = args.gpu
    model.to(device)
    
    epochs = 5
    if args.epochs:
        epochs = args.epochs
    print_every = 32 # Prints every 30 images out of batch of 50 images
    steps = 0
    
    # Train the classifier layers using backpropogation using the pre-trained network to get features
    print("Training process starts .....\n")
    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        running_loss = 0
        model.train() 
        print("epoch: ", epoch)
        for inputs, labels in dataloaders[0]:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()       
            running_loss += loss.item()
            print("step: ", step) 
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
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
        
def test_dataset(model, dataloaders, device, criterion):
    # TODO: Do validation on the test set
    print("Testing process starts .....\n")

    test_loss = 0
    accuracy = 0
    model.eval()

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
    
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Hidden units for DNN classifier')
    
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        type=str,  
                        help='Use GPU or Cuda for calculations')
    
    # Parse args
    args = parser.parse_args()
    return args

def save_checkpoint(model, image_datasets, optimizer, epochs):
    model.class_to_idx = image_datasets.class_to_idx
    checkpoint = {'input_size': 2208,
                  'output_size': 102,
                  'classifier': model.classifier,
                  'optimizer': optimizer,
                  'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,}
    torch.save(checkpoint, 'checkpoint.pth')
    
def load_checkpoint(model, filepath='checkpoint.pth'):
    # Load saved file
    checkpoint = torch.load(filepath)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        model.classifier = checkpoint['classifier']
        optimizer = checkpoint['optimizer']
        epochs = checkpoint['epochs']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']

        
def main():
    """Method to train a new network on a data set in data_directory
        Args: 
            None
        Returns: 
            float: mean of the data set
    """
    
    '''Get argument values'''
    
    args = arg_parser()
    print(args)
        
    data_dir = "flowers"
    if args.data_dir:
        data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_dirs = [train_dir, valid_dir, test_dir]
    
    
    #define transform for the training, validation, and testing sets
    datasets, dataloaders = define_transforms(data_dirs)
    
    model = load_pretrained_network(args.arch)
        
    model = feed_forward(model, args.hidden_units)

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    
    learning_rate = 0.001    
    if args.learning_rate:
        learning_rate = args.learning_rate
        
    epochs = 5
    if args.epochs:
        epochs = args.epochs

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    trainning_network(model, dataloaders, args, optimizer, criterion)
    
    test_dataset(model, dataloaders, device, criterion)
    
    save_checkpoint(model, datasets[0], optimizer, epochs)
    load_checkpoint(model)

if __name__ == '__main__':
    main()
