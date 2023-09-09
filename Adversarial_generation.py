
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import os
import numpy as np
import matplotlib.pylab as plt
import time
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Preprocessing

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

def init_model(lr=0.001, model_name="resnet"):

    if model_name == "resnet":
        model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif model_name == "regnetx":
        model = models.regnet_x_400mf(weights=torchvision.models.RegNet_X_400MF_Weights.DEFAULT)
    elif model_name == "regnety":
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


# Function to load the model
def load_model(model,name="ckpt.pth"):
    global best_acc
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+name)
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return start_epoch, best_acc

# Function to denormalize image
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Denormalize images
mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32)

model, criterion, optimizer, scheduler=init_model(model_name="resnet")
name_model="Resnet.pth"
metrics = load_model(model,name=name_model)
classifier = PyTorchClassifier(model=model,
                               loss=criterion,
                               optimizer=optimizer,nb_classes=10,input_shape=(3,32,32),device_type="gpu")
attack = FastGradientMethod(estimator=classifier,eps=0.05)
adversarial_examples = []
adversarial_labels = []
real_labels= []
for i,data in enumerate(testloader):

    images,labels=data
    
    # images, labels = images.to(device), labels.to(device)
    # # Move the images tensor to CPU before generating adversarial examples
    images_cpu = images.cpu().detach().numpy()
    x_test_adv = attack.generate(x=images_cpu)
    # x_test_adv_gpu = torch.tensor(x_test_adv).to(device)
    # predictions = np.argmax(classifier.predict(x_test_adv_gpu.cpu().detach().numpy()), axis=1)
    with torch.no_grad():
        predictions = np.argmax(classifier.predict(x_test_adv),axis=1)
    # Denormalize the adversarial examples
    x_test_adv_denorm = [denormalize(torch.tensor(x), mean, std) for x in x_test_adv]
    
    adversarial_examples.extend(x_test_adv_denorm)
    real_labels.extend(labels.cpu().numpy())
    adversarial_labels.extend(predictions)


all_adversarial_examples = torch.stack(adversarial_examples)
all_real_labels = torch.tensor(real_labels)
all_adversarial_labels =  torch.tensor(adversarial_labels)

os.makedirs('./Adversarial_examples/FastGradient_Method', exist_ok=True)

adversarial_data={
    'examples': all_adversarial_examples,
    'real_labels': all_real_labels,
    'adversarial_labels': all_adversarial_labels
}

# Save the concatenated adversarial examples and labels
torch.save(adversarial_data, './Adversarial_examples/FastGradient_Method/all_data_denormed.pt')


#### To load the data
# loaded_data = torch.load('./Adversarial_examples/FastGradient_Method/all_data_denorm.pt')
# loaded_adversarial_examples = loaded_data['examples']
# loaded_real_labels = loaded_data['real_labels']
# loaded_adversarial_labels = loaded_data['adversarial_labels']


# Create a TensorDataset from loaded adversarial examples and labels
# from torch.utils.data import TensorDataset, DataLoader

# adversarial_dataset = TensorDataset(loaded_adversarial_examples, loaded_adversarial_labels)

# Create a DataLoader from the TensorDataset
# adversarial_dataloader = DataLoader(adversarial_dataset, batch_size=128, shuffle=False)
##### -------------------------------------------
#### -----------------------
# to perform the predictions we need to have the classifier loaded and then perform


# Now you can use adversarial_dataloader in your training/testing loop


# predictions = np.argmax(classifier.predict(x_test_adv),axis=1)
# images_denorm = denormalize(images[0].cpu().clone(), mean, std)
# x_test_adv_denorm = denormalize(torch.tensor(x_test_adv[0]), mean, std)
# images_denorm = torch.clamp(images_denorm, 0, 1)
# x_test_adv_denorm = torch.clamp(x_test_adv_denorm, 0, 1)

# fig, ax = plt.subplots(2, 1, figsize=(2,5))
# ax[0].imshow(images_denorm.permute(1, 2, 0))
# ax[0].set_title(f"CIFAR label: {labels[0].item()}")
# ax[1].imshow(x_test_adv_denorm.permute(1, 2, 0))
# ax[1].set_title(f"Adversarial example {predictions[0]}:")
# plt.show()