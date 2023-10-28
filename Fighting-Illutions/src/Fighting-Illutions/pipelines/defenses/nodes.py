"""
This is a boilerplate pipeline 'defenses'
generated using Kedro 0.18.13
"""

import matplotlib.pyplot as plt
import matplotlib.figure as matplt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from typing import Tuple, Dict, List

import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def KL_divergence(p,q):
    epsilon = 1e-10
    p += epsilon
    q +=epsilon

    p /= p.sum()
    q /= q.sum()

    return (p*(p/q).log()).sum().item()

def plot_confidence_distribution_single_class(data, class_name,title):
    g = sns.PairGrid(data, diag_sharey=False)
    g.map_upper(sns.kdeplot, levels=12, cmap="icefire")
    g.map_lower(sns.kdeplot, levels=7, cmap="icefire")
    g.map_lower(sns.scatterplot, s=10)
    g.map_diag(sns.histplot, bins=32, stat="density", element="step")
    g.fig.suptitle(f'{class_name} Distributions {title}')
    plt.tight_layout()
    return g.fig  # Return the figure object

def plot_confidence_distribution_all(Confidences, Labels, classes, title):
    figs = []  # List to store individual figures
    Distances = {}
    confidences_model = Confidences["model_confidence"]
    confidences_adversarial = Confidences["adversarial_confidence"]
    confidences_defense = Confidences["defense_confidence"]
    true_labels = Labels["true_labels"]

    for idx, class_name in enumerate(classes):
        values_model = [confidences_model[i] for i, label in enumerate(true_labels) if label == idx]
        values_adversarial = [confidences_adversarial[i] for i, label in enumerate(true_labels) if label == idx]
        values_defense = [confidences_defense[i] for i, label in enumerate(true_labels) if label == idx]

        model_hist = torch.histc(torch.Tensor(values_model),bins=32,min=0,max=1)
        adv_hist = torch.histc(torch.Tensor(values_adversarial),bins=32,min=0,max=1)
        def_hist = torch.histc(torch.Tensor(values_defense),bins=32,min=0,max=1)
        Distances[class_name] = {"Adversarial":KL_divergence(adv_hist,model_hist),"Defense":KL_divergence(def_hist,model_hist)}
        data = pd.DataFrame({"Model": values_model, "Adversarial": values_adversarial, "Defense": values_defense})
        
        fig = plot_confidence_distribution_single_class(data, class_name, title)
        figs.append(fig)
    return figs, Distances
# figs, dists = plot_confidence_distribution_all(confidences,labels,classes,"Regnet_DeepFool_Distortion") 


def plot_sub_confusion_matrix(fig, cm, classes, position, show_colorbar=False):
    """
    This function adds a heatmap of the confusion matrix to a subplot position.
    """
    
    hm = go.Heatmap(
        z=cm[::-1],
        x=classes,
        y=classes[::-1], 
        colorscale='blues',
        showscale=show_colorbar,
        coloraxis="coloraxis",
        hoverinfo="x+y+z",
        hovertemplate="Predicted: %{x}<br>Real: %{y}<br>Number: %{z}<extra></extra>"  # <extra></extra> hides additional hover info
    )

    fig.add_trace(hm, row=1, col=position)

    for i, row in enumerate(cm):
        for j, val in enumerate(row):
             fig.add_annotation(
                dict(
                    x=classes[j], 
                    y=classes[i], 
                    xref=f'x{position}',
                    yref=f'y{position}',
                    text=f"<b>{val}</b>",
                    showarrow=False,
                    font=dict(
                        color="gray",
                        size=12
                    )
                )
            )
    fig.update_yaxes(title_text="True label", row=1, col=position)
    fig.update_xaxes(title_text="Predicted label", row=1, col=position)

def plot_confusion_matrix(labels,classes):
    # Compute confusion matrices
    conf_matrix_defense = confusion_matrix(labels['true_labels'], labels['defense_labels'])
    conf_matrix_model = confusion_matrix(labels['true_labels'], labels['model_labels'])
    conf_matrix_adversarial = confusion_matrix(labels['true_labels'], labels['adversarial_labels'])

    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        subplot_titles=('Model', 'Adversarial', 'Defense'),
        horizontal_spacing=0.01
    )
    plot_sub_confusion_matrix(fig, conf_matrix_model, classes, 1)
    plot_sub_confusion_matrix(fig, conf_matrix_adversarial, classes, 2)
    plot_sub_confusion_matrix(fig, conf_matrix_defense, classes, 3, show_colorbar=True)

    fig.update_layout(title=dict(text="Confusion Matrices", x=0.5, xanchor='center'), height=600, width=1500)
    return fig

#We define the dataset to load the adversarial results
class AdversarialDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        """
        Args:
            data_dict (dict): Dictionary containing adversarial data.
            transform (callable, optional): Optional transform to be applied on the examples.
        """
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_dict["real_labels"])

    def __getitem__(self, idx):
        sample = {key: value[idx] for key, value in self.data_dict.items()}
        if self.transform:
            sample["examples"] = self.transform(sample["examples"])
        return sample

def Create_data_loader(batch_size=128)-> torch.utils.data.DataLoader:

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data/01_raw', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return testloader


def resize_pad(image, ratio=0.8):
    """
    Resizes and pads an image with zeros to match the original size.

    Args:
        image (numpy.ndarray): The input image to resize and pad.
        ratio (float): The ratio to resize the image by (default 0.8).

    Returns:
        torch.Tensor: The resized and padded image.
    """
    original = image.numpy().transpose((1, 2, 0))
    old_size = original.shape[:2]
    new_size = int(old_size[0] * ratio)
    img = cv2.resize(original, (new_size, new_size))
    max_y = old_size[0] - new_size
    max_x = old_size[1] - new_size
    start_y = np.random.randint(0, max_y)
    start_x = np.random.randint(0, max_x)
    pad = np.zeros_like(original)
    pad[start_y:start_y + new_size, start_x:start_x + new_size] = img
    pad = cv2.flip(pad, 1)
    return torch.from_numpy(pad.transpose((2, 0, 1))).float()

def random_distortion(h, w, d, delta):
    """
    Returns distorted coordinates
    """
    nw = w // d
    nh = h // d
    distorted_coords = torch.zeros(nh+1, nw+1, 2)
    
    for m in range(nw+1):
        for n in range(nh+1):
            dx = (torch.rand(1) * 2 - 1) * delta  
            dy = (torch.rand(1) * 2 - 1) * delta 
            x = m * d + dx
            y = n * d + dy
            distorted_coords[n, m, 0] = x
            distorted_coords[n, m, 1] = y
            
    return distorted_coords


def image_distortion(img, d=4, delta=0.5):
    """
    Apply distortion to a given image.
    img: a tensor of shape (C, H, W)
    d: size of the grid
    delta: distortion limit
    """
    C, H, W = img.shape
    nw = W // d
    nh = H // d
    distorted_coords = random_distortion(H, W, d, delta)
    distorted_image = torch.zeros_like(img)
    
    for m in range(nw+1):
        for n in range(nh+1):
            src_x = m * d
            src_y = n * d
            dest_x = int(distorted_coords[n, m, 0].item())
            dest_y = int(distorted_coords[n, m, 1].item())
            for i in range(d+1):
                for j in range(d+1):
                    if src_y + j < H and src_x + i < W and dest_y + j < H and dest_x + i < W:
                        distorted_image[:, dest_y + j, dest_x + i] = img[:, src_y + j, src_x + i]
                        
    return distorted_image
# Here we define the transformations to be used in the dataloader
class ResizePadTransform:
    def __init__(self, ratio=0.8):
        self.ratio = ratio

    def __call__(self, image):
        return resize_pad(image, ratio=self.ratio)
class DistortTransform:
    def __init__(self, d=4, delta=0.5):
        self.d = d
        self.delta = delta
        
    def __call__(self, img):
        return image_distortion(img, self.d, self.delta)



def init_model(model:nn.Module)->torch.nn.Module:
    # Move model to GPU if available
    model = model.to(device)
    if device == 'cuda':
        model= torch.nn.DataParallel(model)
        cudnn.benchmark = True
    return model

def Padding_defense(dataset: Dict, params:Dict) -> torch.utils.data.DataLoader:
    transform_fn = transforms.Compose([
        ResizePadTransform(ratio = params["padding_ratio"]),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    dataset = AdversarialDataset(dataset, transform=transform_fn)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    return dataloader

def Distort_defense(dataset: Dict, params:Dict) -> torch.utils.data.DataLoader:
    transform_fn = transforms.Compose([
        DistortTransform(d = params["window_size"], delta=params["window_size"]),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    dataset = AdversarialDataset(dataset, transform=transform_fn)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    return dataloader

def Padding_Distort_defense(dataset: Dict, params:Dict) -> torch.utils.data.DataLoader:
    transform_fn = transforms.Compose([
        DistortTransform(d=params["window_size"], delta=params["delta"]),
        ResizePadTransform(ratio=params["padding_ratio"]),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = AdversarialDataset(dataset, transform=transform_fn)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    return dataloader


def Report(dataloader:torch.utils.data.DataLoader,model:nn.Module,Report_params:Dict,title:str) -> Tuple[Dict,Dict,go.Figure,List[matplt.Figure]]:
    model_classifier = init_model(model)
    model_classifier.eval()
    correct_defense = 0
    correct_model = 0
    correct_adversarial = 0
    total = 0
    confidence_defense, confidence, confidence_adversarial = [], [], []
    true_labels, model_predictions, adversarial_predictions, defense_predictions = [], [], [], [] 
    
    for batch in dataloader:
        images, real_labels = batch["examples"], batch["real_labels"]
        model_labels, adversarial_labels = batch["model_labels"], batch["adversarial_labels"]
        model_confidence, adversarial_confidence = batch ["confidence"], batch["adversarial_confidence"]

        images, real_labels = images.to(device), real_labels.to(device)
        model_labels, adversarial_labels = model_labels.to(device), adversarial_labels.to(device)
        model_confidence, adversarial_confidence = model_confidence.to(device), adversarial_confidence.to(device) 
        with torch.no_grad():
            outputs = model_classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            confidence_defense.extend(F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy())
            confidence.extend(model_confidence.cpu().numpy())
            confidence_adversarial.extend(adversarial_confidence.cpu().numpy())
            total += real_labels.size(0)
            correct_defense += (predicted == real_labels).sum().item()
            correct_model += (model_labels == real_labels).sum().item()
            correct_adversarial += (adversarial_labels == real_labels).sum().item()
            true_labels.extend(real_labels.cpu().numpy())
            model_predictions.extend(model_labels.cpu().numpy())
            adversarial_predictions.extend(adversarial_labels.cpu().numpy())
            defense_predictions.extend(predicted.cpu().numpy())
            

    original_accuracy = correct_model *100 /total
    adversarial_accuracy = correct_adversarial *100 /total
    defense_accuracy = correct_defense *100 /total

    Accuracies = {"Accuracy":original_accuracy,
                  "Adversarial_accuracy":adversarial_accuracy, 
                  "Defense_accuracy":defense_accuracy}
    
    Labels = {"true_labels": true_labels, 
              "model_labels": model_predictions, 
              "adversarial_labels":adversarial_predictions, 
              "defense_labels":defense_predictions}
    
    Confidences = {"model_confidence":confidence,
                   "adversarial_confidence":confidence_adversarial,
                   "defense_confidence":confidence_defense}
    
    Classes = Report_params["Classes"]
    figs, Kl_dists = plot_confidence_distribution_all(Confidences,Labels,Classes,title) 
    conf_matrix = plot_confusion_matrix(labels=Labels,classes=Classes)
    
    return Accuracies,Kl_dists,conf_matrix ,figs