import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torchvision.transforms as T
from sklearn.metrics import f1_score
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
import glob
#%matplotlib inline

data_dir='../input/food-20/new'
print("The number of classes are:",len(os.listdir(data_dir)))
classes=os.listdir(data_dir)
# The number of classes are: 20

img_dir="../input/food-20/new/*/*.jpg"
number_of_images=len(glob.glob(img_dir))
print(number_of_images)
#20000

image_size=224
batch_size=64
dataset=ImageFolder(data_dir,transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=(-90,90)),
    T.ToTensor()
]))
x,y=dataset[0]
plt.imshow(x.permute(1,2,0))
#<matplotlib.image.AxesImage at 0x7fbb22401d10>

val_size=int(0.2*number_of_images)
train_size=number_of_images-val_size
train_data,val_data=random_split(dataset,[train_size,val_size])
train_loader=DataLoader(train_data,batch_size,shuffle=True,num_workers=3,pin_memory=True)
val_loader=DataLoader(val_data,batch_size*2,num_workers=3,pin_memory=True)

print("The number of samples for training are :",len(train_loader.dataset))
print("The number of samples for training are :",len(val_loader.dataset))
#The number of samples for training are : 16000
#The number of samples for training are : 4000

from torchvision import models
model = models.resnet18(pretrained=True)


for param in model.parameters():
    param.requires_grad = False
from collections import OrderedDict
fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(512,128)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(128,20)),
    ('output', nn.LogSoftmax(dim=1))
]))
model.fc=fc
def get_default_device():
    if torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    return device

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl=dl
        self.device=device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b,self.device)
    def __len__(self):
        return len(self.dl)

device=get_default_device()

train_loader=DeviceDataLoader(train_loader,device)
val_loader=DeviceDataLoader(val_loader,device)
model=to_device(model,device)

NUM_EPOCHS = 10
BEST_MODEL_PATH = 'best_model.pth'
best_loss=100
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    losses=[]
    accs=[]
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    for images, labels in val_loader:
        outputs = model(images)
        pred = torch.argmax(outputs, dim=1)
        acc=torch.tensor(torch.sum(pred==labels).item()/len(labels))
        accs.append(acc)
    acc=torch.stack(accs).mean()
    print("[{}/{}]:\tLoss:{:.4f} Acc:{:.2f}".format(epoch,NUM_EPOCHS-1,losses[epoch],acc.item()))
    if(losses[epoch]<best_loss):
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_loss=losses[epoch]
#[0/9]:	Loss:1.2725 Acc:0.59
#[1/9]:	Loss:1.1890 Acc:0.58
#[2/9]:	Loss:1.3992 Acc:0.59
#[3/9]:	Loss:1.0036 Acc:0.61
#[4/9]:	Loss:1.0288 Acc:0.60
#[5/9]:	Loss:1.3004 Acc:0.61
#[6/9]:	Loss:1.2085 Acc:0.62
#[7/9]:	Loss:1.0235 Acc:0.62
#[8/9]:	Loss:1.2626 Acc:0.61
#[9/9]:	Loss:1.0701 Acc:0.61