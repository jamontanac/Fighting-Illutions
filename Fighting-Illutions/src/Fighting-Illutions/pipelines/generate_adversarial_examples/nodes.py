"""
This is a boilerplate pipeline 'generate_adversarial_examples'
generated using Kedro 0.18.13
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import art
import numpy as np
from art.attacks.evasion import FastGradientMethod, DeepFool
from art.estimators.classification import PyTorchClassifier

from typing import Tuple,Dict, List,Any
import logging
import importlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Create_data_loader()-> torch.utils.data.DataLoader:

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data/01_raw', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)
    return testloader


def init_model(model:torch.nn.Module,lr:float=0.001)->Tuple[nn.Module,nn.Module,optim.Optimizer]:
    # Move model to GPU if available
    model = model.to(device)
    if device == 'cuda':
        model= torch.nn.DataParallel(model)
        cudnn.benchmark = True
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return model, criterion, optimizer

def denormalize(tensor, mean= torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32), std =  torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32)):
    denorm = torch.clone(tensor)
    for t, m, s in zip(denorm, mean, std):
        t.mul_(s).add_(m)
    return denorm


def normalize(tensor, mean= torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32), std =  torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32)):
    norm = torch.clone(tensor)
    for t, m, s in zip(norm, mean, std):
        t.sub_(m).div_(s)
    return norm


def classification(model:nn.Module)-> art.estimators.classification.pytorch.PyTorchClassifier:
    model, criterion, optimizer = init_model(model)

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
    return classifier
def Evasion_Attack(classifier:art.estimators.classification.pytorch.PyTorchClassifier,attack:Dict):
    attack_module = attack["module"]
    attack_type = attack["class"]
    attack_arguments = attack["kwargs"]
    
    attack_class=getattr(importlib.import_module(attack_module),attack_type)
    attack_instance = attack_class(classifier,**attack_arguments)


    return attack_instance

def Adversarial_generation(classifier, attack_params: Dict):
    attack = Evasion_Attack(classifier,attack_params)
    logger = logging.getLogger(__name__)

    logger.info(f"Creating attack of type {type(attack)}")
    testloader = Create_data_loader()
    real_labels = []
    model_labels = []
    adversarial_examples = []
    adversarial_labels = []
    for data in testloader:
        images, labels = data
        images_cpu = images.cpu().detach().numpy()
        x_test_adv = attack.generate(x=images_cpu,y=labels.cpu().numpy())
        with torch.no_grad():
            predictions = np.argmax(classifier.predict(images_cpu),axis=1)
            adversarial_predictions = np.argmax(classifier.predict(x_test_adv),axis=1)
        
        adversarial_denorm = [denormalize(torch.tensor(x)) for x in x_test_adv]

        adversarial_examples.extend(adversarial_denorm)
        real_labels.extend(labels.cpu().numpy())
        model_labels.extend(predictions)
        adversarial_labels.extend(adversarial_predictions)
        
    all_adversarial_examples = torch.stack(adversarial_examples)
    all_real_labels = torch.tensor(real_labels)
    all_model_labels = torch.tensor(model_labels)
    all_adversarial_labels = torch.tensor(adversarial_labels)

    adversarial_data = {
        "examples": all_adversarial_examples,
        "real_labels": all_real_labels,
        "model_labels": all_model_labels,
        "adversarial_labels": all_adversarial_labels,
    }

    old_accuracy = (all_real_labels == all_model_labels).sum().item()/ all_real_labels.size(0)
    new_accuracy =  (all_real_labels == all_adversarial_labels).sum().item()/ all_real_labels.size(0)
    logger.info(f"Accuracy of the model was {old_accuracy*100:.2f}% and now is {new_accuracy*100:.2f}%")
    return adversarial_data


def Fast_gradient_attack(classifier:art.estimators.classification.pytorch.PyTorchClassifier,attackparam:Dict)->Dict[str,Any]:
    testloader = Create_data_loader()
    real_labels = []
    model_labels = []
    aux=0
    logger = logging.getLogger(__name__)
    for eps in attackparam["eps"]:
        logger.info(f"Starting to do FSGM with strength {eps}")
        adversarial_examples = []
        adversarial_labels = []
        attack = FastGradientMethod(estimator=classifier,eps=eps)
        for data in testloader:
            images,labels = data

            images_cpu = images.cpu().detach().numpy()
            x_test_adv = attack.generate(x=images_cpu)
            if aux==0:

                with torch.no_grad():
                    predictions = np.argmax(classifier.predict(images_cpu),axis=1)
                    adversarial_predictions = np.argmax(classifier.predict(x_test_adv),axis=1)

                adversarial_denorm = [denormalize(torch.tensor(x)) for x in x_test_adv]
                # we keep the adversarial examples to use them later
                adversarial_examples.extend(adversarial_denorm)
                real_labels.extend(labels.cpu().numpy())
                model_labels.extend(predictions)
                adversarial_labels.extend(adversarial_predictions)
            else:

                with torch.no_grad():
                    adversarial_predictions = np.argmax(classifier.predict(x_test_adv),axis=1)
                adversarial_denorm = [denormalize(torch.tensor(x)) for x in x_test_adv]
                adversarial_examples.extend(adversarial_denorm)
        if aux==0:

            all_adversarial_examples = torch.stack(adversarial_examples)
            all_real_labels = torch.tensor(real_labels)
            all_model_labels = torch.tensor(model_labels)
            all_adversarial_labels = torch.tensor(adversarial_labels)

            adversarial_data = {
                f"examples_{eps}": all_adversarial_examples,
                "real_labels": all_real_labels,
                "model_labels": all_model_labels,
                f"adversarial_labels_{eps}": all_adversarial_labels,
            }
        else:
            all_adversarial_examples = torch.stack(adversarial_examples)
            all_adversarial_labels = torch.tensor(adversarial_labels)

            adversarial_data[f"examples_{eps}"] = all_adversarial_examples
            adversarial_data[f"adversarial_labels_{eps}"] = all_adversarial_labels
        aux+=1

    return adversarial_data
