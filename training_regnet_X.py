
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import os
import matplotlib.pylab as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

def init_model(lr=0.001):
    # lr=0.005
    model = models.regnet_x_400mf(weights=torchvision.models.RegNet_X_400MF_Weights.DEFAULT)
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


# Function to save the model
def save_model(epoch, best_acc,model,name="ckpt.pth"):
    print('Saving..')
    state = {
        'model': model.state_dict(),
        'acc': best_acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+name)


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
# Training function
def train(epoch,model,optimizer, criterion):
    model.train()
    correct = 0
    total = 0
    train_loss = 0
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
            print(f"[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}| Accuracy: {100*correct/total:.3f} ({correct} / {total})")
    acc= 100* correct/total 
    return train_loss/total, acc
# Test function
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
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
    print(f"Accuracy of the network on the {total} test images: {acc}%")
    return test_loss/total, acc  # Add this line to return the test accuracy

train_loss_hist = []
test_loss_hist  = []
train_acc_hist  = []
test_acc_hist  = []
# Initialize best_acc and start_epoch
model, criterion, optimizer, scheduler = init_model()

# Uncomment to load the model
name_model="Regnet_X.pth"
best_acc = 0
start_epoch = 0
if os.path.exists("./checkpoint/"+name_model):
    start_epoch, best_acc = load_model(model,name=name_model)

epochs = range(start_epoch,start_epoch+30)
# Main loop
for epoch in epochs:
    train_loss,train_acc = train(epoch,model,optimizer, criterion)
    test_loss, test_acc = test(model)  # Modify test() to return the accuracy
    scheduler.step()
    train_loss_hist.append(train_loss)
    train_acc_hist.append(test_acc)
    test_loss_hist.append(test_loss)
    test_acc_hist.append(test_acc)

    # Save the model if test accuracy is greater than best_acc
    if test_acc > best_acc:
        save_model(epoch, test_acc,model, name=name_model)
        best_acc = test_acc

print("Finished Training")
# reporting the model in the specific run
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(epochs,train_loss_hist, label = "Training Loss")
plt.plot(epochs,test_loss_hist, label = "Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.subplot(1,2,2)
plt.plot(epochs,train_acc_hist, label = "Training Accuracy")
plt.plot(epochs,test_acc_hist, label = "Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.tight_layout()
plt.savefig("Result_training"+name_model[:-4]+str(time.time()).split(".")[0]+".png")
plt.close()