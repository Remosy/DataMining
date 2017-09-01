import gensim
from os import listdir
from os.path import isfile, join

class PreProcess():

    cleaned_Data=""

    def __init__(self):
        super().__init__()
        self.cleanData(self.loadData())
        self.vectorData()

    def loadData(self,folderPath):
        amazon = [];
        imdb = [];
        yelp = [];
        #List and save all fiels from the folder
        for file in listdir(folderPath):
            if file.endswith(".txt",".csv"):
                print(join(folderPath, file))


    def vectorData(self,cleaned_Data):
        text2vector = gensim.models.Word2Vec.load(cleaned_Data)