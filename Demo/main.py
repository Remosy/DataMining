from preprocess import PreProcess
#from textMiner import TextMiner
class TextMining():
    def __init__(self):
        super().__init__()
        self.folderPath = "/Users/remosy/DataMining/Demo/textData"
        self.preProcess = PreProcess()

        # without transforming string
        self.lemmatizedList = self.preProcess.load_data1(self.folderPath)
        self. allWords = (" ".join(self.lemmatizedList))
        print(self.allWords)
        # transform lemmatized_list(3 generators) into a long string separate by space
        self.sparseWords =  self.preProcess.cleanByFrequency(self.allWords)

        # filter by sequence
        self.cleanedSentences = self.preProcess.getCleanedSent(self.sparseWords,self.lemmatizedList)

        self.displayData()


    def displayData(self):
        #for i in self.vectors[0]:
            #print(i)
        #print("File 1 has: "+ str(sum(1 for x in (self.lemmatized_1[0]))))
        #print("File 2 has: " + str(sum(1 for x in (self.lemmatized_1[1]))))
        #print("File 3 has: " + str(sum(1 for x in (self.lemmatized_1[2])))+'\n')
        print("[After lemmatizer, there are "+str(self.lemmatizedList.__len__())+" sentences]")


        #print("All text has: " + str(len(self.lemmatized_2.split())))
        print("After filter, the # of words(freq <= 5) has: "+ str(len(self.sparseWords))+'\n')
        print("Un-common dictionary = ")
        print(self.sparseWords)
        print(self.cleanedSentences)

        print("Finished \n")

if __name__ == '__main__':
    textMining = TextMining()