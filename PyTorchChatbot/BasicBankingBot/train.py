import json

from torch.cuda import is_available
from model import NeuralNet

from nltk import tag
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

from preprocess import stem,tokenize, bag_of_words

with open('intents.json','r') as f:
    intents = json.load(f)

tags = []
all_words = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        pattern_tokens = tokenize(pattern)
        all_words.extend(pattern_tokens)
        xy.append((pattern_tokens,tag))

ignore_words = ["!","?",".",",",":",")"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#print(tags)
#print(all_words)
#print(xy)

X_train = []
y_train = []

for query_tokens, tag in xy:
    bag = bag_of_words(query_tokens,all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples

#Hyperparameters
batch_size = 8
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size).to(device)