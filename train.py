### Imports

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
import IPython.display as display
import argparse




###Parser

# Setting up arguments for parser
parser = argparse.ArgumentParser (description = "Training script parser")

parser.add_argument('data_dir', help = 'Input data directory. Mandatory argument', type = str)
parser.add_argument('--save_dir', help = 'Input destination directory for saving.', type = str)

parser.add_argument('--arch', help = 'Architecture, VGG11 is the default architecture. VGG13 is set as an option.', type = str)

parser.add_argument('--learning_rate', help = 'Learning rate, default value is 0.001.', type = float)
parser.add_argument('--hidden_units', help = 'Hidden units in Classifier, default value is 500', type = int)
parser.add_argument('--epochs', help = 'Number of epochs, default value is 2', type = int)
parser.add_argument('--GPU', help = "Option to use GPU. Input either True/False. default value is True", type = str)

#Setting up parser
args = parser.parse_args()



### Setting up Variables

#Saving directory
if args.save_dir:
    save_dir = args.save_dir + '/checkpoint.pth'
else:
    save_dir = "checkpoint.pth"

#Defining device
if args.GPU == 'False':
    user_device = 'cpu'
else:
    user_device = 'cuda'

#Choosing the model
if args.arch == 'VGG13':
    arch = args.arch
    model = models.vgg13(pretrained=True)
else:
    arch = 'VGG11'
    model = models.vgg11(pretrained=True)

#Hidden Layers
if args.hidden_units:
    hidden_units = args.hidden_units
else:
    hidden_units = 500

#Epochs
if args.epochs:
    epochs = args.epochs
else:
    epochs = 2

#Learning Rate
if args.learning_rate:
    lrn = args.learning_rate
else:
    lrn = 0.001

    
### Data Loading
    
#Setting up directories
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#data loading
#making sure the value is supplied
if data_dir: 
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

#Label Mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


###Building the Network

device = torch.device(user_device)
print(device, "is being used")

# Freeze parameters so we don't backprop through them

for param in model.parameters():
    param.requires_grad = False

# Use classifier with 102 outputs which are trained.
classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

model.classifier = classifier
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen

optimizer = optim.Adam(model.classifier.parameters(), lr=lrn)
model.to(device)  
    

    
## Training the Network

print('Training has started...')
      
epochs = epochs
steps = 0
running_loss = 0
print_every = 20

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1

        # Move input and label tensors to the default device for trainloader
        inputs, labels = inputs.to(device), labels.to(device)

        #Reset optimizer
        optimizer.zero_grad()

        #Run the model
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        #Calculate Training Loss
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:

                    #Move input and label tensors to the default device for validloader
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    #Calculate Testing Loss
                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Steps {steps}.. "
                  f"Train loss: {running_loss/print_every:.2f}.. "
                  f"Validation loss: {test_loss/len(validloader):.2f}.. "
                  f"Validation accuracy: {(accuracy/len(validloader))*100:.2f}")
            running_loss = 0
            model.train()

print('training is complete')

      
#Validation on the test set

def accuracy_on_test(testloader):    
    correct = 0
    total = 0
    
    #Move model to GPU
    model.to('cuda:0')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            
            #Run the test set through the model
            outputs = model(images)
            
            #Get the class with the highest probability
            _, predicted = torch.max(outputs.data, 1)
            
            # Add up all the images in the test set
            total += labels.size(0)
            
            #Add up all the correctly classified images
            correct += (predicted == labels).sum().item()
            
    #Print the accuracy which should be atleast >70%
    print('Accuracy of model on test images: %d %%' % ((correct / total)*100))
    
accuracy_on_test(testloader)


# Saving the checkpoint 

checkpoint = {'arch': arch,
              'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'class_to_idx': train_data.class_to_idx,
              'opt_state': optimizer.state_dict,
              'num_epochs': epochs}

torch.save(checkpoint, save_dir)
print('checkpoint has been saved at:', save_dir)