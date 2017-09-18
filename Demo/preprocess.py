import string
from os import listdir
from os.path import join

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize, word_tokenize
from nltk import WordNetLemmatizer
from nltk import FreqDist
from nltk import pos_tag


class PreProcess:
    def __init__(self,stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords or set(sw.words('english'))
        self.punct = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def load_data1(self, folder_path):
        collection = []
        for file in listdir(folder_path):
            if file.endswith(".txt") | file.endswith(".csv"):
                path = join(folder_path, file)
                print("Start to process file: " + path, end='\n')
                collection.append(self.process_doc(path))
        return collection

    def load_data2(self, folder_path):
        collection = ""
        for file in listdir(folder_path):
            if file.endswith(".txt") | file.endswith(".csv"):
                path = join(folder_path, file)
                print("Start to process file: " + path, end='\n')
                temp = " ".join(self.process_doc(path))
                collection += (temp + " ")
        return collection

    def process_doc(self, path):
        #print("Start to process file: " + path, end='\n')
        count = 0
        f = open(path, 'r')
        for line in f:
            line = line.split('\t')
            count = count + 1
            #print("No. "+str(count)+": "+line[0], end='\n')
            for token, tag in pos_tag(wordpunct_tokenize(line[0])):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token
                #token = token.strip('.') if self.strip else token
                #token = token.strip(',') if self.strip else token

                #print("[" + token + "____" + tag + " ] ")

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

                # Debug using
                #for i in lemma:
                    #print("Lemmatize: " + i, end='\n')
                #print("--------------------------------")
                #if count > 3:
                    #print("***")
                    #return

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        return self.lemmatizer.lemmatize(token, tag)

    def transform(self, X):
        return [" ".join(doc) for doc in X]

    def cleanByFrequency(self,lemmatized_string):
        #Filter by the frequency of appearance <= 5
        token = word_tokenize(lemmatized_string)
        distance = FreqDist(token)
        return list(filter(lambda x: x[1] <= 5, distance.items()))

    def vector_Data(self, cleaned_Data):
        return
