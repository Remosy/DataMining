
from gensim.models import KeyedVectors, Doc2Vec

from documentLevel import DocumentLevel
from preprocess import PreProcess
#from textMiner import TextMiner
from sentenceLevel import SentenceLevel



class TextMining():
    def __init__(self):
        super().__init__()
        self.folderPath = "/Users/remosy/DataMining/Demo/textData"

        # without transforming string
        # List structure: [[data1][data1_label][data2][data2_label][data3][data3_label]]
        #self.lemmatizedList = self.preProcess.load_data1(self.folderPath)
        #Extract data
        #self.dataList = self.getDataList()
        #print(self.dataList.__len__())
        #self.allWords = self.extractSentence()
        #print(self.allWords.__len__())
        #Extract label
        #self.Labels = self.extractLabels()
        #print(self.Labels.__len__())
        # transform lemmatized_list(3 generators) into a long string separate by space
        #self.sparseWords =  self.preProcess.cleanByFrequency(self.allWords)
        # filter by sequence
        #self.cleanedSentences = self.preProcess.getCleanedSent(self.sparseWords,self.dataList)
        '''Pre Process'''
        self.preProcess = PreProcess(self.folderPath)

        # vectorise remnant of sentences
        #self.X_train = self.preProcess.vector_Data(self.cleanedSentences,self.Labels)
        '''Process Sentence Level'''
        self.sentenceLevel = SentenceLevel(self.preProcess.cleanedSentences)

        '''Process Document Level'''
        # Start TensorFlow Session
        self.documentLevel = DocumentLevel(self.sentenceLevel.docInput,self.preProcess.Labels)

        # display results
        #self.displayData()
        '''Summary All Process'''


    def displayData(self):
        #for i in self.vectors[0]:
            #print(i)
        #print("File 1 has: "+ str(sum(1 for x in (self.lemmatized_1[0]))))
        #print("File 2 has: " + str(sum(1 for x in (self.lemmatized_1[1]))))
        #print("File 3 has: " + str(sum(1 for x in (self.lemmatized_1[2])))+'\n')
        #print("[After lemmatizer, there are "+str(self.lemmatizedList.__len__())+" sentences]")


        #print("All text has: " + str(len(self.lemmatized_2.split())))
        #print("After filter, the # of words(freq <= 5) has: "+ str(len(self.sparseWords))+'\n')
        print("Un-common dictionary = ")
        #print(self.sparseWords)
        #print(self.cleanedSentences)

        #Vectorise
        word_vectors = Doc2Vec.load('textData/wordsVectors.doc2vec')  # text format
        print(word_vectors.docvecs)
        # self.plotWords()
        print("Finished \n")

    def plotWords(self):
        
       return

if __name__ == '__main__':
    textMining = TextMining()