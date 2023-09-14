"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.13
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
# import os
# import matplotlib.pylab as plt
# import time
from typing import Tuple,Dict, List
import logging
# import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def Create_data_loader()-> torch.utils.data.DataLoader:
    # Data Preprocessing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) # these are the parameter for CIFAR10

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data/01_raw', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data/01_raw', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)
    return trainloader,testloader
def init_model(lr:float=0.001,model_name:str="Resnet")->Tuple[torch.nn.Module,torch.nn.Module,torch.optim.Optimizer,torch.optim.lr_scheduler.LRScheduler]:
    if model_name == "Resnet":
        model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif model_name == "Regnet_x":
        model = models.regnet_x_400mf(weights=torchvision.models.RegNet_X_400MF_Weights.DEFAULT)
    elif model_name == "Regnet_y":
        model = models.regnet_y_400mf(weights=torchvision.models.RegNet_Y_400MF_Weights.DEFAULT)
    # Fine-tuning: Replace the last layer (classifier)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    # Move model to GPU if available
    model = model.to(device)
    if device == 'cuda':
        model= torch.nn.DataParallel(model)
        cudnn.benchmark = True
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return model, criterion, optimizer, scheduler


# # Function to save the model
# def save_model(epoch, best_acc,model,name="ckpt.pth"):
#     print('Saving..')
#     state = {
#         'model': model.state_dict(),
#         'acc': best_acc,
#         'epoch': epoch,
#     }
#     if not os.path.isdir('checkpoint'):
#         os.mkdir('checkpoint')
#     torch.save(state, './data/06_models'+name)


# # Function to load the model
# def load_model(model,name="ckpt.pth"):
#     global best_acc
#     logger = logging.getLogger(__name__)
#     logger.info('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/'+name)
#     model.load_state_dict(checkpoint['model'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
#     return start_epoch, best_acc
    
# Training function
def train(epoch:int,model:torch.nn.Module,optimizer:torch.optim.Optimizer, criterion:torch.nn.Module)-> Tuple[float,float]:
    logger = logging.getLogger(__name__)
    model.train()
    correct = 0
    total = 0
    train_loss = 0
    trainloader, testloader = Create_data_loader()
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 100 == 99:
            logger.info(f"[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}| Accuracy: {100*correct/total:.3f} ({correct} / {total})")
            # print(f"[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}| Accuracy: {100*correct/total:.3f} ({correct} / {total})")
    acc= 100* correct/total 
    return train_loss/total, acc
# Test function
def test(model:torch.nn.Module,criterion:torch.nn.Module) -> Tuple[float,float]:
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    logger = logging.getLogger(__name__)
    trainloader, testloader = Create_data_loader()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    logger.info(f"Accuracy of the network on the {total} test images: {acc}%")
    # print(f"Accuracy of the network on the {total} test images: {acc}%")
    return test_loss/total, acc  # Add this line to return the test accuracy

def Train_model(parameters:Dict)->Tuple[torch.nn.Module,pd.Series,pd.Series, pd.Series, pd.Series]:
    train_loss_hist = []
    test_loss_hist  = []
    train_acc_hist  = []
    test_acc_hist  = []
    name_model = parameters["model_name"]
    model, criterion, optimizer, scheduler = init_model(model_name=name_model)
    best_acc = 0

    logger = logging.getLogger(__name__)
    epochs = range(0,parameters["epochs"])
    # Main loop
    for epoch in epochs:
        train_loss,train_acc = train(epoch,model,optimizer, criterion)
        test_loss, test_acc = test(model,criterion)
        scheduler.step()
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

        # Save the model if test accuracy is greater than best_acc
        if test_acc > best_acc:
            best_model = model
            # save_model(epoch, test_acc,model, name=name_model)
            best_acc = test_acc
            best_epoch = epoch
        logger.info(f"Best model so far has {best_acc} and was on the epoch {best_epoch}")
    return best_model,pd.Series(train_loss_hist,name="Train Loss"),pd.Series(train_acc_hist,name="Train Accuracy"), pd.Series(test_loss_hist,name="Test Loss"), pd.Series(test_acc_hist,name="Test Accuracy")


# print("Finished Training")
# # reporting the model in the specific run
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.plot(epochs,train_loss_hist, label = "Training Loss")
# plt.plot(epochs,test_loss_hist, label = "Test Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")

# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(epochs,train_acc_hist, label = "Training Accuracy")
# plt.plot(epochs,test_acc_hist, label = "Test Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")

# plt.legend()
# plt.tight_layout()
# plt.savefig("Result_training"+name_model[:-4]+str(time.time()).split(".")[0]+".png")
# plt.close()