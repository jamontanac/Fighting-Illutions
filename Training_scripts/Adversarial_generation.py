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
print(device)
# Data Preprocessing
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Data Loading
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)


# Function to init the model
def init_model(lr=0.001, model_name="Resnet"):
    if model_name == "Resnet":
        model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif model_name == "Regnet_X":
        model = models.regnet_x_400mf(
            weights=torchvision.models.RegNet_X_400MF_Weights.DEFAULT
        )
    elif model_name == "Regnet_Y":
        model = models.regnet_y_400mf(
            weights=torchvision.models.RegNet_Y_400MF_Weights.DEFAULT
        )

    # Fine-tuning: Replace the last layer (classifier)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    # Move model to GPU if available
    model = model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return model, criterion, optimizer, scheduler


# Function to load the model
def load_model(model, name="ckpt.pth"):
    global best_acc
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    if device == "cuda":
        checkpoint = torch.load("./checkpoint/" + name)
    else:
        checkpoint = torch.load(
            "./checkpoint/" + name, map_location=torch.device("cpu")
        )
    model.load_state_dict(checkpoint["model"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    return start_epoch, best_acc


# Function to denormalize image
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# Denormalize images
mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32)

names = ["Resnet.pth", "Regnet_X.pth", "Regnet_Y.pth"]

for name_model in names:
    model, criterion, optimizer, scheduler = init_model(model_name=name_model[:-4])
    metrics = load_model(model, name=name_model)
    if device == "cuda":
        device_type = "gpu"
    else:
        device_type = "cpu"
    
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        nb_classes=10,
        input_shape=(3, 32, 32),
        device_type=device_type,
    )

    # attack = FastGradientMethod(estimator=classifier, eps=0.01, eps_step= 0.001, norm="inf", minimal= True, batch_size=128,targeted=False)
    attack = FastGradientMethod(estimator=classifier, eps=0.01, norm="inf", batch_size=128,targeted=False)
    # print(type(attack))
    adversarial_examples = []
    adversarial_labels = []
    real_labels = []
    model_labels = []
    for i, data in enumerate(testloader):
        images, labels = data

        # # Move the images tensor to CPU before generating adversarial examples
        images_cpu = images.cpu().detach().numpy()
        x_test_adv = attack.generate(x=images_cpu,y=labels.cpu().numpy())
        with torch.no_grad():
            predictions = np.argmax(classifier.predict(images_cpu), axis=1)
            adver_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)

        # Denormalize the adversarial examples
        x_test_adv_denorm = [
            denormalize(torch.tensor(x), mean, std) for x in x_test_adv
        ]

        adversarial_examples.extend(x_test_adv_denorm)
        real_labels.extend(labels.cpu().numpy())
        model_labels.extend(predictions)
        adversarial_labels.extend(adver_predictions)

    all_adversarial_examples = torch.stack(adversarial_examples)
    all_real_labels = torch.tensor(real_labels)
    all_model_labels = torch.tensor(model_labels)
    all_adversarial_labels = torch.tensor(adversarial_labels)

    os.makedirs("./Adversarial_examples/FastGradient_Method", exist_ok=True)

    adversarial_data = {
        "examples": all_adversarial_examples,
        "real_labels": all_real_labels,
        "model_labels": all_model_labels,
        "adversarial_labels": all_adversarial_labels,
    }

    # Save the concatenated adversarial examples and labels
    torch.save(
        adversarial_data,
        f"./Adversarial_examples/FastGradient_Method/all_data_denormed_{name_model[:-4]}.pt",
    )
    old_accuracy = (all_real_labels == all_model_labels).sum().item()/ all_real_labels.size(0)
    new_accuracy =  (all_real_labels == all_adversarial_labels).sum().item()/ all_real_labels.size(0)
    print(f"Accuracy of the model {name_model} was {old_accuracy*100:.3f}% and now is {new_accuracy*100:.3f}%")
    print(attack.attack_params[0])