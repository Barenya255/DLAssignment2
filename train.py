''' the following will take in command line arguments and do the same thing that the pynb was doing.'''

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import requests
import zipfile
import os
from math import ceil as ceil
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb

import argparse

''' The following lines add the argparsing feature to take in user input.
    all values need to be give as is, either with short hands or not.
    The entity and project name are made to mine.
    I have no idea what is to be done with the wandb key for login() purposes.
'''

parser = argparse.ArgumentParser(description='Training the model as per arguments passed')
parser.add_argument('-lg', '--log', type = str, default = "no")
parser.add_argument('-ld', '--load', type = str, default = "NIL", help = "enter name of model to load into and place the model in the same directory.")
parser.add_argument('-wp', '--wandb_project', type = str, default = "DLAssignment2", help = "Default has been set to my project name. Please change as per required")
parser.add_argument('-we', '--wandb_entity', type = str, default = "cs22m028")
parser.add_argument('-e', '--epochs',type = int, default = 20)
parser.add_argument('-b', '--batchSize',type = int, default = 32)
parser.add_argument('-f', '--factor',type = int, default = 2)
parser.add_argument('-lr', '--learningRate', type = float, default = 1e-4)
parser.add_argument('-w_d', '--weight_decay', type = float, default = 0)
parser.add_argument('-d', '--dropOut', type = float, default = 0.5)
parser.add_argument('-k', '--kernels', type = str, default = "5 5 3 3 3", help = "Enter 5 space separated integers")
parser.add_argument('-fs', '--filterSize', type = int, default = 32, help = "Enter 5 space separated integers")
parser.add_argument('-a', '--activation', type = str, default = "mish", help = "can choose from only mish, silu, gelu and relu")
parser.add_argument('-pre', '--pretrained', type = str, default = "no", help = "train from scratch or from pretrained weights")
parser.add_argument('-aug', '--augmentation', type = str, default = "yes", help = " augment data or not")
parser.add_argument('-bN', '--batchNorm', type = str, default = "yes", help = " augment data or not")


args = parser.parse_args()

if args.batchNorm == "yes":
    args.batchNorm = True
else:
    args.batchNorm = False

kernel = args.kernels.split()

new_kernel =[]

for element in kernel:
    new_kernel.append(int(element))

args.kernels = new_kernel

################################################################################################

################################################################################################
# Prepare the dataset for being pushed into the training pipeling.
def prepData(augment:bool):
    ''' Function to prepare the data using torch libraries for the purpose of training torch
        neural networks with relative ease.
        
        Using torch dataLoaders helps in memory management as well
        
        args : augment bool ---> True would enable data augmentation, False would disable.
        
        return : 
            TrainDataLoader --> torch data loader wrapper for training dataset.
            ValDataLoader ----> torch data loader wrapper for validation data set.
            TestDataLoader ---> torch data loader wrapper for test data set.'''
    
    if augment == True:
        preProcess = transforms.Compose([
            transforms.Resize(size = (128,128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor()
        ])
    else:
        preProcess = transforms.Compose([
            transforms.Resize(size = (128,128)),
            transforms.ToTensor()
        ])
    print("loading the data into tensors ==============================")
    trainData = datasets.ImageFolder(root = "nature_12K/inaturalist_12K/train",
                                    transform = preProcess,
                                    target_transform = None)
    classLabels = trainData.classes
    
    testData  = datasets.ImageFolder(root = "nature_12K/inaturalist_12K/val",
                                    transform = preProcess)

    print(f"train data : {trainData} and test data : {testData}")

    print("splitting into train and val ================================")
    trainSplit = ceil(0.8*len(trainData))
    trainData, valData = torch.utils.data.random_split(trainData, [trainSplit, len(trainData) - trainSplit])

    print("wrapping into train loader ==================================")

    trainDataLoader = torch.utils.data.DataLoader(trainData,
                                                shuffle=True,
                                                batch_size=32)

    valDataLoader = torch.utils.data.DataLoader(valData,
                                                shuffle=True,
                                                batch_size=32)

    testDataLoader = torch.utils.data.DataLoader(testData,
                                                shuffle=False,
                                                batch_size=32)
    
    print("loaders created for faster loading ===========================")



    return trainDataLoader, valDataLoader, testDataLoader, classLabels

##############################################################################################################################################

##############################################################################################################################################

def prepDataForModel(preProcess):
    ''' Function to prepare the data using torch libraries for the purpose of training torch
        neural networks with relative ease.
        
        Using torch dataLoaders helps in memory management as well
        
        args : augment bool ---> True would enable data augmentation, False would disable.
        
        return : 
            TrainDataLoader --> torch data loader wrapper for training dataset.
            ValDataLoader ----> torch data loader wrapper for validation data set.
            TestDataLoader ---> torch data loader wrapper for test data set.'''
    
    print("loading the data into tensors ==============================")
    trainData = datasets.ImageFolder(root = "nature_12K/inaturalist_12K/train",
                                    transform = preProcess,
                                    target_transform = None)
    classLabels = trainData.classes
    
    testData  = datasets.ImageFolder(root = "nature_12K/inaturalist_12K/val",
                                    transform = preProcess)

    print(f"train data : {trainData} and test data : {testData}")

    print("splitting into train and val ================================")
    trainSplit = ceil(0.8*len(trainData))
    trainData, valData = torch.utils.data.random_split(trainData, [trainSplit, len(trainData) - trainSplit])

    print("wrapping into train loader ==================================")

    trainDataLoader = torch.utils.data.DataLoader(trainData,
                                                shuffle=True,
                                                batch_size=32)

    valDataLoader = torch.utils.data.DataLoader(valData,
                                                shuffle=True,
                                                batch_size=32)

    testDataLoader = torch.utils.data.DataLoader(testData,
                                                shuffle=False,
                                                batch_size=32)
    
    print("loaders created for faster loading ===========================")



    return trainDataLoader, valDataLoader, testDataLoader, classLabels



############################################################################################

############################################################################################
# Defininition of the mode
class CNNModel(nn.Module):
    ''' CNN Model for classifying the images
    
        __init__ : creates a blueprint for the model
        forward  : forward propagation facilitated by Torch Layers.'''

    def __init__(self, activation, kernels, inputShape: int, hiddenUnit: int, outputSize: int, dropOut: float, batchNorm: bool, factor: int):
        
        ''' initialize the model == inherit from nn.Module
            
            args : activation --> activation Function torch.nn.$SomeValidActivationFunction$
                   kernels ---> list conatining 5 kernel sizes that may be taken as input.
                   inputShape --> Number of Channels in input data.
                   hiddenUnit --> Filter size.
                   outputSize --> number of output channels.
                   batchNorm ---> boolean var to indicate whether to add batch normalization or not.
                   factor ------> int value used as multiplier for subsequent layers.
        '''
        super().__init__()
        self.hiddenUnit = hiddenUnit
        self.factor = factor
        self.batchNorm = batchNorm

        self.conv_blocks = nn.ModuleList()    # ModuleList (Torch container) used to record specific layers.  
        self.batch_norms = nn.ModuleList()    # Using ModuleList shortens boiler plate code.

        layerSize = [inputShape] + [self.hiddenUnit] + [self.factor * self.hiddenUnit] * 4
        for i in range(5):
            self.conv_blocks.append(nn.Conv2d(layerSize[i], layerSize[i+1], kernel_size=kernels[i], padding=2))  # add conv layers
            if self.batchNorm:
                self.batch_norms.append(nn.BatchNorm2d(layerSize[i+1])) # add Batch normalization only if specified.

        self.activate = activation  # add activation function (taken as input).
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # max pool layer (as and when to be used).
        self.drop = nn.Dropout(p=dropOut)  # drop out layer for reducing over fitting.
        
        # DenseBlock containing a flattening layer, dense layer.
        self.DenseBlock = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=1024, bias=True, device=None, dtype=None), # LazyLinear used for calculation of in_features.
            nn.Dropout(p=dropOut),
            nn.Linear(in_features=1024, out_features=outputSize)
        )

    def forward(self, x):
        ''' Forward Propagation
            x ---> tensor denoting input value.
            return x ---> prediction value.'''
        for i in range(5):
            x = self.conv_blocks[i](x)
            if self.batchNorm and i < 5:
                x = self.batch_norms[i](x)
            x = self.activate(x)
            x = self.pool(x)

        x = self.drop(x)
        x = self.DenseBlock(x)
        x = nn.functional.softmax(x, dim=1)

        return x

################################################################################################

################################################################################################


def accuracy(y_true, y_pred):
    ''' accuracy Function for calculating the percentage of y_true[i] == y_pred[i]
    args : y_true ---> int actual value/ label(s) of for the input(s).
    return : accuracy ---> float [0,100] The accuracy of the batch.
    '''
    correct = torch.eq(y_true,y_pred).sum().item()
    accuracy = 0.0
    accuracy = correct/(len(y_true))*100
    return accuracy

################################################################################################

################################################################################################

# To train the model.
def fit(model, trainDataLoader, valDataLoader, epochs, device, loss_fn, optimizer):
    ''' Function for training the model on the data set.
    args --->
        model -> CNNModule object 
        trainDataLoader --> torch dataLoader wrapper containing training set.
        valDataLoader --> torch dataLoader wrapper containing validation set.
        epochs --> int, number of epochs.
        device --> whether cpu or cuda.
        loss_fn ---> loss Function used.
        optimizer --> optimizer function used.
        
    return model --> CNN Module object with updated weights.
    '''
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        train_acc = 0
        for batch, (X,y) in enumerate(trainDataLoader):
            X,y = X.to(device), y.to(device)
            model.train()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += accuracy(y_true=y, y_pred=y_pred.argmax(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch%50 == 0:
                print(f"went through {batch}/{len(trainDataLoader)} samples")
            torch.cuda.empty_cache()

        train_loss /= len(trainDataLoader)
        train_acc /= len(trainDataLoader)

        val_loss = 0.0
        val_acc = 0
        model.eval()
        with torch.inference_mode():
            for X,y in valDataLoader:
                X,y = X.to(device), y.to(device)
                val_pred = model(X)
                val_loss += loss_fn(val_pred, y)
                val_acc += accuracy(y_true=y, y_pred=val_pred.argmax(dim=1))
            val_acc /= len(valDataLoader)
            val_loss /= len(valDataLoader)

        wandb.log({"TrainingLoss" : train_loss, "ValidationLoss" : val_loss, "TrainingAccuracy" : train_acc, "ValidationAccuracy" : val_acc, "epoch": epoch})

        print(f"Train loss: {train_loss}, Train accuracy: {train_acc}, validation loss: {val_loss}, validation accuracy: {val_acc}\n")

    return model

#################################################################################################

################################################################################################

# Fucnction for evaluating on test data (unseen)

def eval(model, testDataLoader, device, loss_fn):
    ''' Function for evaluating the training on unseen test Dataset.
        args --> testDataLoader torch DataLoader object for easy loading/unloading.
    '''
    test_loss = 0.0
    test_acc = 0
    model.eval()
    with torch.inference_mode():
        for X,y in testDataLoader:
            X,y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy(y_true=y, y_pred=test_pred.argmax(dim=1))
        test_acc /= len(testDataLoader)
        test_loss /= len(testDataLoader)
    print(f"Test Accuracy is {test_acc} and Test Loss os {test_loss}")

################################################################################################

################################################################################################

# This function takes care of the all parameters needed for training.

def masterTrainer(trainDataLoader, valDataLoader, testDataLoader, learningRate, kernels, layerSize, dropOut, batchNorm, activation, factor, epochs):
    ''' function to start the training and facilitate wandb logging.
        args ->
            trainDataLoader --> torch dataLoader wrapper containing training set.
            valDataLoader --> torch dataLoader wrapper containing validation set.
            testDataLoader --> torch dataLoader wrapper containing test dataset.
            
            learningRate ---> int, learning rate,
            kernels --> list of kernel sizes, one for each convolutional layer.
            layerSize --> filter size of first convolutional layer
            factor --> multiplier to number of filters for subsequent training.
            epochs --> int, number of epochs.
    '''
    activations = {
    "relu" : torch.nn.ReLU(),
    "gelu" : torch.nn.GELU(),
    "silu" : torch.nn.SiLU(),
    "mish" : torch.nn.Mish()
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #import wandb    
    #wandb.init(project="DLAssignment2", entity="cs22m028")
    #wandb.run.name = "config_"+str(optimizer)+"_"+str(layerSize)+"_"+str(decay)+"_"+str(opt)+"_"+str(batchNorm)+"_"+str(dropOut)+"_"+str(activation)    
    activate= activations[activation]
    if args.pretrained == "no":
        print(" Building model from scratch...")
        model_0 = CNNModel(activate, kernels, inputShape=3, hiddenUnit=layerSize,outputSize=10, dropOut = dropOut, batchNorm=True, factor = factor)
        print(" Model ready ")
    else:
        print(" Training on vision transformer pretrained model...")
        weights = models.ViT_B_16_Weights.DEFAULT
        auto_transforms = weights.transforms()

        # device selection code. If GPU available, choose it, else stick to CPU.

        #import the model.
        model_0 = models.vit_b_16(weights=weights).to(device)

        #Freeze the parameters.
        for params in model_0.parameters():
            params.requires_grad=False

        # Modify the last layer so as to fit the output space.
        lastLayer = model_0.heads.head.in_features

        # add Dropout so as to prevent overFitting.
        model_0.heads.head = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(lastLayer,10))
        trainDataLoader, valDataLoader, testDataLoader, classLabels = prepDataForModel(auto_transforms)

        print(" Vision Transformer pretrained model ready... ")

    model_0.to(device)
    #from helper_functions import accuracy_fn as accuracy # Note: could also use torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model_0.parameters(),lr=learningRate)
    print(model_0)
    fit(model_0, trainDataLoader, valDataLoader, epochs, device, loss_fn, optimizer)

    print("########################################################################################################")
    print("############################ successfully trained the model ############################################")
    print("########################################################################################################")

    print()
    print()

    print("########################################################################################################")
    eval(model_0, testDataLoader, device, loss_fn)
    print("########################################################################################################")


################################################################################################

################################################################################################

# Function for logging into wandb.
def wandbTrainer():
    ''' wandb trainer for initializing runs, loading data and running the entire training/ testing process.'''
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    wandb.run.name="while_args_config_activation="+str(args.activation)+"_epochs="+str(args.epochs)+"_dropOut="+str(args.dropOut)+"_batchSize="+str(args.batchSize) + "_filterSize="+str(args.filterSize)+"_batchNorm="+str(args.batchNorm)+"_augment="+str(args.augmentation)+"_learningRate="+str(args.learningRate)
    trainDataLoader, valDataLoader, testDataLoader, classLabels = prepData(augment=args.augmentation)
    masterTrainer(trainDataLoader, valDataLoader, testDataLoader, args.learningRate, args.kernels, args.filterSize, args.dropOut, args.batchNorm, args.activation, args.factor, args.epochs)
    #masterTrainer(trainDataLoader, valDataLoader, testDataLoader, config.wandb.learningRate, config.wandb.decay, config.wandb.kernels, config.wandb.layerSize, config.wandb.dropOut, config.wandb.batchNorm, config.wandb.activation, config.wandb.factor)

################################################################################################

################################################################################################

def main():
    trainDataLoader, valDataLoader, testDataLoader, classLabels = prepData(augment=args.augmentation)
    loss_fn = nn.CrossEntropyLoss()

    masterTrainer(trainDataLoader, valDataLoader, testDataLoader, args.learningRate, args.kernels, args.filterSize, args.dropOut, args.batchNorm, args.activation, args.factor, args.epochs)
    
def extract():
    with zipfile.ZipFile("nature_12K.zip", "r") as extraction:
        extraction.extractall("nature_12K")

def download():
    file = "nature_12K.zip"

    if os.path.exists(file) == True:
        print("File exists, proceeding to extraction....")

        if os.path.exists("nature_12K"):
            print("Already extracted. If extraction corrupted, please delete the folder")
        else:
            extract()

    else:
        with open("nature_12K.zip", "wb") as f:
            print("download has begun")
            request = requests.get("https://storage.googleapis.com/wandb_datasets/nature_12K.zip")
            f.write(request.content)
            print("successful download")
            print("################################################################################################")
            extract()

def loadAndEvalModel(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(path, map_location=torch.device(device))
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    trainDataLoader, valDataLoader, testDataLoader, classLabels = prepData(augment=args.augmentation)
    eval(model, testDataLoader, device, loss_fn)

if __name__ == '__main__':
    if args.load != "NIL":
        download()
        loadAndEvalModel(args.load)
    else:
        print("downloading data....")
        download()
        if args.log != "no":
            key1 = input("enter wandb key: ")
            wandb.login(key=key1)
            wandbTrainer()
        else:
            print("################################################################################################")
            print("####################### starting the training process ##########################################")
            print("################################################################################################")
            main()