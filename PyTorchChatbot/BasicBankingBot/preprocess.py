import nltk
from nltk.stem import PorterStemmer

nltk.download("punkt")
stemmer = PorterStemmer()

def tokenize(query):
    return nltk.word_tokenize(query)

def stem(word):
    return stemmer.stem(word.lower())


if __name__=="__main__":
    query = "Hello! How are you doing?"
    tokenized_query = tokenize(query)
    print(tokenized_query)
    stemmed_tokens = [stem(w) for w in tokenized_query]
    print(stemmed_tokens)
