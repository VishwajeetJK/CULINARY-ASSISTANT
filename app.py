import base64
import numpy as np
import io
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from flask import request
from flask import jsonify
from flask import Flask
from foodrecom1 import *
import csv

classes=['Cheesecake','Chicken_Curry','Chicken_Wings','Chocolate_Cake','Chocolate_Mousse','Cup_Cakes','Fre nch_Fries',
'French_Toast','Fried_Rice','Garlic_Bread','Ice_Cream','Macaroni_And_Cheese','Nachos','Omelette','Pancakes', 'Pizza','Samosa','Spring_Rolls','Strawberry_Shortcake','Waffles']
rec={}
with open('FOOD RECOMMENDATION SYSTEM-3.csv', mode='r') as infile:
    reader = csv.reader(infile)
    rec = {rows[0]:[rows[1],rows[2]] for rows in reader}

app = Flask(__name__)
model = models.resnet18(pretrained=False)
fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(512,128)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(128,20)), ('output', nn.LogSoftmax(dim=1))
]))

model.fc=fc
model.load_state_dict(torch.load('/Users/Vishwajeet/Documents/AI/transfer/best_model-66.pth',map_location=torch.device('cpu'))) model.eval()
def transforms(image_bytes):
    transform=T.Compose([ T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    return transform(image_bytes).unsqueeze(0)

def get_prediction(image_tensor):
    outputs=model(image_tensor)
    _,pred=torch.max(outputs.data,1)
    return pred

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    image = image.convert("RGB")
    image_tensor = transforms(image)
    prediction = get_prediction(image_tensor)
    pred=classes[prediction.item()].replace("_"," ")
    data={"prediction":pred}
    recommendation=my_food_recommendation(pred)
    data['rec1']=recommendation[0]
    data['rec2']=recommendation[1]
    data['rec']=rec[pred][0]
    data['ing']=rec[pred][1]
    data['img1']=recommendation[2]
    data['img2']=recommendation[3]
    return jsonify(data)

if __name__=="__main__":
    app.run(debug=True)