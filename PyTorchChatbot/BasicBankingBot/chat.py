import random
import json

import torch

from model import NeuralNet
from preprocess import bag_of_words,tokenize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("intents.json", "r") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
f = torch.load(FILE)

input_size = f["input_size"]
output_size = f["output_size"]
hidden_size = f["hidden_size"]
all_words = f["all_words"]
tags = f["tags"]
model_state = f["model_state"]

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Budbak"
print("Let's chat! Type 'quit' to exit")

while(True):
    sentence = input("You: ")
    if(sentence == "quit"):
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _,predicted = torch.max(output,dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]
    if(prob.item()>0.75):
        for intent in intents["intents"]:
            if(tag == intent["tag"]):
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")