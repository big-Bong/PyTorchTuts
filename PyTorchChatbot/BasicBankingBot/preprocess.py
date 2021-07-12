import enum
import nltk
from nltk.stem import PorterStemmer
import numpy as np

nltk.download("punkt")
stemmer = PorterStemmer()

def tokenize(query):
    return nltk.word_tokenize(query)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_query,all_words):
    stemmed_tokens = [stem(w) for w in tokenized_query]

    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx,word in enumerate(all_words):
        if(word in stemmed_tokens):
            bag[idx] = 1.0
    
    return bag


if __name__=="__main__":
    query = "Hello! How are you doing?"
    tokenized_query = tokenize(query)
    print(tokenized_query)
    stemmed_tokens = [stem(w) for w in tokenized_query]
    print(stemmed_tokens)

    sentence = ["hello","how","are","you"]
    words = ["hi","hello","I","you","bye","thank","cool"]
    bag = bag_of_words(sentence,words)

    print(bag)
