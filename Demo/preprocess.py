import string
from os import listdir
from os.path import join
from time import strftime, gmtime

from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize, word_tokenize
from nltk import WordNetLemmatizer
from nltk import FreqDist
from nltk import pos_tag

import gensim, logging


class PreProcess:
    def __init__(self,folderPath,stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords or set(sw.words('english'))
        self.punct = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        ############# Process ############
        self.lemmatizedList = self.load_data1(folderPath)
        self.dataList = self.getDataList()
        self.allWords = self.extractSentence()
        self.Labels = self.extractLabels()
        self.sparseWords = self.cleanByFrequency(self.allWords)
        self.cleanedSentences = self.getCleanedSent(self.sparseWords, self.dataList)

    def load_data1(self, folder_path):
        collection = []
        for file in listdir(folder_path):
            if file.endswith(".txt") | file.endswith(".csv"):
                path = join(folder_path, file)
                print("Start to process file: " + path, end='\n')
                collection+=(self.process_doc(path))
                print(collection.__len__())
        return collection

    def process_doc(self, path):
        #print("Start to process file: " + path, end='\n')
        count = 0
        f = open(path, 'r')
        doc = []
        labels = []
        for line in f:
            line = line.split('\t')
            count = count + 1
            #print("No. "+str(count)+": "+line[0], end='\n')
            tmp_str = " ".join(self.processSentence(line))
            doc.append(tmp_str)
            labels.append(line[1])
        #print(doc)
        return [doc,labels]


    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        return self.lemmatizer.lemmatize(token, tag)

    def processSentence(self,line):
        for token, tag in pos_tag(wordpunct_tokenize(line[0])):
            # Apply preprocessing to the token
            token = token.lower() if self.lower else token
            token = token.strip() if self.strip else token
            token = token.strip('_') if self.strip else token
            token = token.strip('*') if self.strip else token

            # print("[" + token + "____" + tag + " ] ")

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
            # for i in lemma:
            # print("Lemmatize: " + i, end='\n')
            # print("--------------------------------")
            # if count > 3:
            # print("***")
            # return

    def transform(self, X):
        return [" ".join(doc) for doc in X]

    def cleanByFrequency(self,lemmatized_string):
        #Filter by the frequency of appearance <= 5
        token = word_tokenize(lemmatized_string)
        distance = FreqDist(token)
        sparses = list(filter(lambda x: x[1] <= 5, distance.items()))
        return [i[0] for i in sparses]

    def getCleanedSent(self,sparseWords,lemmatized_strings):
        lemmatized_list = []
        for x in lemmatized_strings:
            list = x.split(' ')
            lemmatized_list.append([j for j in list if j not in sparseWords])
        print(lemmatized_list)
        return lemmatized_list

    def vector_Data(self, cleaned_Data):
        model = Word2Vec(cleaned_Data, size=100, window=2, min_count= 1)
        print(model)
        model.init_sims(replace=True)
        timeStamp = strftime("%d%b_%H_%M_%S", gmtime())
        model.wv.save_word2vec_format('Output/WV'+timeStamp+'.word2vec.txt', binary=False)
        model.wv.save_word2vec_format('Output/WV'+timeStamp+'.word2vec.bin', binary=True)
        return model

    def extractSentence(self):
        allLabels = " "
        allLabels += " ".join(self.lemmatizedList[0])
        allLabels += " ".join(self.lemmatizedList[2])
        allLabels += " ".join(self.lemmatizedList[4])
        return allLabels

    def extractLabels(self):
        list = []
        list+=(self.lemmatizedList[1])
        list+=(self.lemmatizedList[3])
        list+=(self.lemmatizedList[5])
        return list

    def getDataList(self):
        list = []
        list+=(self.lemmatizedList[0])
        list+=(self.lemmatizedList[2])
        list+=(self.lemmatizedList[4])
        return list


    # Method support doc2vec
    def makePreVector(self,cleaned_Data,labels):
        for x, y in zip(cleaned_Data,labels):
            yield TaggedDocument(x,[y])
