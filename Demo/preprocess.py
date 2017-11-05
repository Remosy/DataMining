import string
from os import listdir
from os.path import join
from time import strftime, gmtime

from gensim.models import Word2Vec
import numpy as np
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
            l = line[1][0]
            if int(l) != 0 | int(l) != 1:
                raise Warning("Label is wrong!")
            else:
                labels.append(int(l))
        #print(doc)
        return [doc,labels]

    def lemmatize(self, token, tag):
        tags = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ

        }
        if tag.startswith('N'):
            return self.lemmatizer.lemmatize(token, wn.ADJ)
        elif tag.startswith('N'):
            return self.lemmatizer.lemmatize(token, wn.VERB)
        elif tag.startswith('N'):
            return self.lemmatizer.lemmatize(token, wn.ADV)
        else:
            return self.lemmatizer.lemmatize(token)




    def processSentence(self,line):
        for token, tag in pos_tag(word_tokenize(line[0])):
            #print("[Before: " + token + "____" + tag + " ] ")
            # Apply preprocessing to the token
            token = token.lower() if self.lower else token
            token = token.strip() if self.strip else token
            token = token.strip('_') if self.strip else token
            token = token.strip('*') if self.strip else token

            # If punctuation, ignore token and continue
            if all(char in self.punct for char in token):
                continue

            print("[before: " + token + "____" + tag + " ] ")

            if token == "'d":
                token = "could"

            if token == "'ll":
                token = "will"

            if token == "'ve":
                token = "have"

            if token == "'s" and tag=="VBZ":
                token = "has"

            if token == "'s" and tag=="POS":
                token = "is"

            if token == "n't":
                token = "not"

            if token == "'re":
                token = "are"
            #print("[" + token + "____" + tag + " ] ")
            # If stopword, ignore token and continue
            #if tag in ("PRP","PRP$","IN","TO",'WDT',"DT","CC","EX","WP","WRB"):
            #    continue

            if tag not in ("JJ","JJR","JJS","MD","RB","RBR","RBS"):
                print("[after: " +  "....... ____" + tag + " ] ")
                continue
            #if token in self.stopwords:
                #continue

            print("[After" + token + "____" + tag + " ] ")

            # Lemmatize the token and yield
            lemma = self.lemmatize(token, tag)
            yield lemma
            token = lemma

            #print("[after: " + token + "____" + tag + " ] ")
            # Debug using
            #for i in lemma:
             #print("Lemmatize: " + i, end='\n')
             #print("--------------------------------")
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
        print("labels size = " + str(len(list)))
        list_np = np.zeros((3000,1),dtype=int)
        for i in np.arange(len(list)):
            list_np[i] = list[i]
        return list_np

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
