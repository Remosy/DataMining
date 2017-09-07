import string
from os import listdir
from os.path import join

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag


class PreProcess:
    def __init__(self, folder_path, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords or set(sw.words('english'))
        self.punct = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        rawSentences = []
        self.load_data(folder_path)
        # self.vectorData()

    def load_data(self, folder_path):
        for file in listdir(folder_path):
            if file.endswith(".txt") | file.endswith(".csv"):
                path = join(folder_path, file)
                self.process_doc(path)
        return self

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def process_doc(self, path):
        print("Start to process file: " + path, end='\n')
        count = 0
        f = open(path, 'r')
        for line in f:
            line = line.split('\t')
            count = count + 1
            print("No. "+str(count)+": "+line[0], end='\n')
            for token, tag in pos_tag(wordpunct_tokenize(line[0])):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token
                #token = token.strip('.') if self.strip else token
                #token = token.strip(',') if self.strip else token

                print("[" + token + "____" + tag + " ] ")

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                #yield lemma

                for i in lemma:
                    print("Lemmatize: " + i, end='\n')
                print("--------------------------------")
                if count > 3:
                    print("***")
                    return


                


    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        return self.lemmatizer.lemmatize(token, tag)

    def vector_Data(self, cleaned_Data):
        return
