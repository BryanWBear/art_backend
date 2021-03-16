from PIL import Image
from io import BytesIO
from flask import Flask, abort, request, make_response
import base64
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import json

class ResNet18(nn.Module):
    def __init__(self, num_classes, dropout_prob):
        super().__init__()

        self.model_ft = models.resnet18(pretrained=True)
        self.model_ft.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_prob/2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.model_ft(x)
        x = self.fc(x)
        return x

app = Flask(__name__)

model = ResNet18(3, 0.5)
model.load_state_dict(torch.load('res.h5'))
model.eval()

transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

@app.route('/ping')
def ping():
    return 'server is up'


artist_map = {0: 'Edgar Alwin Payne', 1: 'Norman Rockwell', 2: 'Pablo Picasso'}

@app.route('/api/v1/predict/', methods = ['POST'])
def predict():
    image = request.json['img']
    image = Image.open(BytesIO(base64.b64decode(image)))
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        pred = model(image)
    pred = list(torch.softmax(pred.squeeze(), -1).numpy())
    idx_max = np.argmax(pred)
    print(f"Prediction: {artist_map[idx_max]}, Confidence: {round(pred[idx_max].item(), 2)}")
    return f"Prediction: {artist_map[idx_max]}, Confidence: {round(pred[idx_max], 2)}"

app.run(host='0.0.0.0')

