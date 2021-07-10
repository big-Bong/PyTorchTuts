import json

from nltk import tag
from preprocess import stem,tokenize

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

print(tags)
print(all_words)
print(xy)